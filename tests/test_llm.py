import asyncio
import json
import logging
import time
from email.utils import formatdate

import httpx
import pytest

from app.config import Settings
from app.errors import ExtractionError
from app.llm import (
    ALLOWED_MODELS,
    CooldownGate,
    LLMBadOutput,
    LLMClient,
    LLMError,
    LLMUnavailable,
    Usage,
    parse_json_reply,
    parse_retry_after,
)

MODEL = "google/gemini-3.8-flash"
MESSAGES = [{"role": "user", "content": "SECRET DOCUMENT TEXT"}]


def settings(**kw):
    defaults = dict(llm_base_url="http://gw.test/v1/", llm_api_key="sk-test-key", llm_backoff_base_seconds=0.01,
                    llm_backoff_max_seconds=0.03, llm_call_budget_seconds=5)
    return Settings(**{**defaults, **kw})


def ok(content="hello", finish="stop", usage=None, headers=None):
    body = {"choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": usage if usage is not None else {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001}}
    return httpx.Response(200, json=body, headers=headers)


class Script:
    """A fake gateway that replays responses in order, then repeats the last one."""

    def __init__(self, *steps):
        self.steps, self.requests = list(steps), []

    def __call__(self, request):
        self.requests.append(request)
        step = self.steps.pop(0) if len(self.steps) > 1 else self.steps[0]
        if isinstance(step, Exception):
            raise step
        return step


def run(script, body=None, **kw):
    async def main():
        client = LLMClient(settings(**kw), transport=httpx.MockTransport(script))
        usage = Usage()
        try:
            outcome = await (body(client, usage) if body else client.chat(MODEL, MESSAGES, usage))
        except ExtractionError as exc:
            outcome = exc
        finally:
            await client.close()
        return outcome, usage

    return asyncio.run(main())


def test_uses_configured_endpoint_and_key():
    s = Script(ok())
    result, _ = run(s, llm_base_url="http://gateway.example/api/v1/", llm_api_key=" my-key ")
    assert result.content == "hello"
    request = s.requests[0]
    assert str(request.url) == "http://gateway.example/api/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer my-key"
    assert json.loads(request.content)["model"] == MODEL


def test_every_listed_model_is_allowed_and_nothing_else():
    assert len(ALLOWED_MODELS) == 9
    s = Script(ok())

    async def body(client, usage):
        return await client.chat("openai/gpt-4o", MESSAGES, usage)

    result, _ = run(s, body)
    assert isinstance(result, LLMError) and result.code == "llm_model_not_allowed"
    assert s.requests == []  # refused before any request is sent


@pytest.mark.parametrize("base,key", [("", "k"), ("http://x", ""), ("  ", "  ")])
def test_missing_configuration_fails_clearly(base, key):
    s = Script(ok())
    result, _ = run(s, llm_base_url=base, llm_api_key=key)
    assert result.code == "llm_not_configured" and s.requests == []


@pytest.mark.parametrize("failure", [
    httpx.Response(429), httpx.Response(408), httpx.Response(409), httpx.Response(500), httpx.Response(502),
    httpx.Response(503), httpx.Response(504), httpx.Response(529),
    httpx.ConnectError("refused"), httpx.ReadTimeout("slow"), httpx.ConnectTimeout("slow"),
    httpx.RemoteProtocolError("dropped"), httpx.ReadError("reset"),
    httpx.Response(200, json={"error": {"code": 429, "message": "x"}}),
    httpx.Response(200, json={"error": {"code": 502, "message": "x"}}),
    httpx.Response(200, json={"error": {"code": "503", "message": "x"}}),
    httpx.Response(200, json={"error": {"message": "no code at all"}}),
], ids=str)
def test_transient_failures_are_retried_until_success(failure):
    s = Script(failure, failure, ok())
    result, usage = run(s)
    assert result.content == "hello" and len(s.requests) == 3
    assert usage.as_metadata()["model_calls"] == 3  # retries are counted


@pytest.mark.parametrize("status,code", [
    (400, "llm_rejected"), (404, "llm_rejected"), (422, "llm_rejected"),
    (401, "llm_auth_failed"), (403, "llm_auth_failed"), (402, "llm_out_of_credits"),
])
def test_permanent_errors_fail_without_retrying(status, code):
    s = Script(httpx.Response(status, text="provider said SECRET"))
    result, _ = run(s)
    assert result.code == code and len(s.requests) == 1
    assert "SECRET" not in result.message


def test_permanent_error_inside_a_200_body():
    s = Script(httpx.Response(200, json={"error": {"code": 401, "message": "bad key"}}))
    result, _ = run(s)
    assert result.code == "llm_auth_failed" and len(s.requests) == 1


@pytest.mark.parametrize("reply", [
    httpx.Response(200, text="<html>not json</html>"), httpx.Response(200, json=[1, 2]), httpx.Response(200, json={}),
    httpx.Response(200, json={"choices": []}), httpx.Response(200, json={"choices": [None]}),
    httpx.Response(200, json={"choices": [{"message": {"content": None}}]}),
    httpx.Response(200, json={"choices": [{"message": {"content": "   "}}]}),
], ids=lambda r: r.text[:30])
def test_unusable_replies_get_a_limited_number_of_retries(reply):
    s = Script(reply)
    result, _ = run(s)
    assert isinstance(result, LLMBadOutput) and len(s.requests) == 3


def test_unusable_reply_can_recover():
    s = Script(httpx.Response(200, json={}), ok("fine"))
    assert run(s)[0].content == "fine"


def test_retries_stop_when_the_budget_runs_out():
    s = Script(httpx.Response(503))
    start = time.monotonic()
    result, usage = run(s, llm_call_budget_seconds=0.15)
    assert isinstance(result, LLMUnavailable) and result.code == "llm_unavailable"
    assert time.monotonic() - start < 1
    assert usage.as_metadata()["model_calls"] >= 2


def test_long_retry_after_fails_fast_instead_of_sleeping_past_the_budget():
    s = Script(httpx.Response(429, headers={"Retry-After": "3600"}))
    start = time.monotonic()
    result, _ = run(s, llm_call_budget_seconds=1)
    assert isinstance(result, LLMUnavailable) and time.monotonic() - start < 0.5


def test_retry_after_is_respected():
    s = Script(httpx.Response(429, headers={"Retry-After": "0.3"}), ok())
    start = time.monotonic()
    assert run(s)[0].content == "hello"
    assert time.monotonic() - start >= 0.3


def test_budget_override_per_call():
    s = Script(httpx.Response(503))

    async def body(client, usage):
        return await client.chat(MODEL, MESSAGES, usage, budget_seconds=0.05)

    assert isinstance(run(s, body, llm_call_budget_seconds=60)[0], LLMUnavailable)


def test_cancellation_stops_retrying():
    s = Script(httpx.Response(503))

    async def main():
        client = LLMClient(settings(llm_backoff_base_seconds=0.05, llm_backoff_max_seconds=0.05),
                           transport=httpx.MockTransport(s))
        task = asyncio.create_task(client.chat(MODEL, MESSAGES, Usage()))
        await asyncio.sleep(0.12)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        count = len(s.requests)
        await asyncio.sleep(0.15)
        assert len(s.requests) == count
        await client.close()

    asyncio.run(main())


def test_usage_totals_across_models_and_bad_usage_values():
    s = Script(ok(usage={"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.5}),
               ok(usage={"prompt_tokens": "x", "completion_tokens": -5, "cost": None}), ok(usage=[]))

    async def body(client, usage):
        await client.chat(MODEL, MESSAGES, usage)
        await client.chat("deepseek/deepseek-v3.2", MESSAGES, usage)
        await client.chat("deepseek/deepseek-v3.2", MESSAGES, usage)
        return usage.as_metadata()

    meta, _ = run(s, body)
    assert (meta["model_calls"], meta["tokens_in"], meta["tokens_out"], meta["cost_usd"]) == (3, 100, 20, 0.5)
    assert meta["models"][MODEL]["calls"] == 1 and meta["models"]["deepseek/deepseek-v3.2"]["calls"] == 2


def test_key_and_document_never_reach_logs_or_errors(caplog):
    s = Script(httpx.Response(503), httpx.Response(401))
    with caplog.at_level(logging.DEBUG):
        result, _ = run(s)
    text = caplog.text + str(result)
    assert "sk-test-key" not in text and "SECRET DOCUMENT TEXT" not in text


@pytest.mark.parametrize("text,expected", [
    ('{"a": 1}', {"a": 1}), ('```json\n{"a": 1}\n```', {"a": 1}), ('```\n[1,2]\n```', [1, 2]),
    ('Here it is:\n{"a": 1}\nThanks!', {"a": 1}), ('[{"a": 1}]', [{"a": 1}]),
    # Raw tab copied from a DOCX table cell, as seen in a real reply for lease_002.
    ('{"quote": "April 30, 2027\t$2,845.00"}', {"quote": "April 30, 2027\t$2,845.00"}),
])
def test_json_replies_are_parsed(text, expected):
    assert parse_json_reply(text) == expected


@pytest.mark.parametrize("text", ["no json", '{"a": ', '{"a": NaN}', ""])
def test_invalid_json_replies_raise(text):
    with pytest.raises(ValueError):
        parse_json_reply(text)


def test_chat_json_retries_bad_json_then_succeeds():
    s = Script(ok("not json"), ok('{"a": 1}'))

    async def body(client, usage):
        return await client.chat_json(MODEL, MESSAGES, usage)

    assert run(s, body)[0] == {"a": 1} and len(s.requests) == 2


def test_chat_json_gives_up_on_repeated_bad_json():
    s = Script(ok("not json"))

    async def body(client, usage):
        return await client.chat_json(MODEL, MESSAGES, usage)

    assert isinstance(run(s, body)[0], LLMBadOutput) and len(s.requests) == 3


def test_chat_json_treats_cut_off_replies_as_unusable():
    s = Script(ok('{"a": 1}', finish="length"), ok('{"a": 2}'))

    async def body(client, usage):
        return await client.chat_json(MODEL, MESSAGES, usage)

    assert run(s, body)[0] == {"a": 2}
    assert json.loads(s.requests[1].content)["max_tokens"] == 32_000  # retried with more room


def test_cut_off_reply_is_retried_once_with_more_room_then_given_up():
    s = Script(ok('{"a": 1', finish="length"))

    async def body(client, usage):
        return await client.chat_json(MODEL, MESSAGES, usage, max_tokens=8000)

    assert isinstance(run(s, body)[0], LLMBadOutput) and len(s.requests) == 2
    assert [json.loads(r.content)["max_tokens"] for r in s.requests] == [8000, 16000]


def test_parse_retry_after_formats():
    now = 1_800_000_000.0
    def parse(**h):
        return parse_retry_after(httpx.Headers({k.replace("_", "-"): v for k, v in h.items()}), now=now)
    assert parse(retry_after="7") == 7
    assert parse(retry_after="1.5") == 1.5
    assert parse(retry_after="-3") == 0
    assert parse(retry_after=formatdate(now + 30, usegmt=True)) == pytest.approx(30, abs=1)
    assert parse(retry_after=formatdate(now - 30, usegmt=True)) == 0
    assert parse(x_ratelimit_reset=str(int((now + 20) * 1000))) == pytest.approx(20)  # milliseconds
    assert parse(x_ratelimit_reset=str(int(now + 20))) == pytest.approx(20)  # seconds
    assert parse(retry_after="soon") is None and parse(x_ratelimit_reset="abc") is None and parse() is None
    assert parse(retry_after="inf") is None and parse(retry_after="nan") is None


def test_real_sockets_with_dropped_connections_and_garbage():
    """A real HTTP server that misbehaves in ways a mock transport can't imitate."""
    count = {"n": 0}

    async def handle(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        count["n"] += 1
        n = count["n"]
        if n == 1:  # drop the connection without responding
            writer.transport.abort()
            return
        if n == 2:  # not HTTP at all
            writer.write(b"garbage\r\n\r\n")
        elif n == 3:  # headers promise more body than arrives
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 500\r\n\r\n{\"choi")
        elif n == 4:
            writer.write(b"HTTP/1.1 429 Too Many\r\nRetry-After: 0.05\r\nContent-Length: 0\r\n\r\n")
        else:
            body = json.dumps({"choices": [{"message": {"content": "survived"}, "finish_reason": "stop"}]}).encode()
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s" % (len(body), body))
        await writer.drain()
        writer.close()

    async def main():
        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        client = LLMClient(settings(llm_base_url=f"http://127.0.0.1:{port}/v1"))
        try:
            return await client.chat(MODEL, MESSAGES, Usage())
        finally:
            await client.close()
            server.close()

    assert asyncio.run(main()).content == "survived"
    assert count["n"] == 5


# ----- Shared cooldown, against real Postgres -----

def test_one_workers_429_makes_other_workers_wait(with_store):
    async def body(db, store):
        s = Script(httpx.Response(429, headers={"Retry-After": "0.4"}), ok())
        first = LLMClient(settings(), CooldownGate(db), transport=httpx.MockTransport(s))
        other_requests = []

        def other(request):
            other_requests.append(time.monotonic())
            return ok()

        second = LLMClient(settings(), CooldownGate(db), transport=httpx.MockTransport(other))
        try:
            task = asyncio.create_task(first.chat(MODEL, MESSAGES, Usage()))
            await asyncio.sleep(0.1)  # the first client has now been rate limited
            start = time.monotonic()
            await second.chat(MODEL, MESSAGES, Usage())
            assert other_requests[0] - start >= 0.2  # it held back without being throttled itself
            await task
        finally:
            await first.close()
            await second.close()

    with_store(body)


def test_cooldown_is_per_model(with_store):
    async def body(db, store):
        await CooldownGate(db).report(MODEL, 30)
        gate = CooldownGate(db)
        assert await gate.seconds_remaining(MODEL) > 25
        assert await gate.seconds_remaining("deepseek/deepseek-v3.2") == 0

    with_store(body)


def test_shorter_report_never_shortens_a_cooldown(with_store):
    async def body(db, store):
        gate = CooldownGate(db)
        await gate.report(MODEL, 30)
        await CooldownGate(db).report(MODEL, 1)
        assert await CooldownGate(db).seconds_remaining(MODEL) > 25

    with_store(body)


def test_gate_still_works_locally_when_the_database_is_down():
    from app.db import PostgresDatabase

    async def main():
        db = PostgresDatabase("postgresql://u:p@127.0.0.1:1/d", acquire_timeout_seconds=0.1)
        await db.open()
        gate = CooldownGate(db)
        await gate.report(MODEL, 10)
        assert await gate.seconds_remaining(MODEL) > 5
        await db.close()

    asyncio.run(main())
