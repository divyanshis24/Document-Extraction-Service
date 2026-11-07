"""The only path to a model. Handles throttling, provider failures and unusable replies.

Provider trouble is retried until the caller's time budget runs out, because a rate limit
says nothing about the document. Errors that retrying can't fix fail immediately.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import httpx
import psycopg
from psycopg_pool import PoolClosed, PoolTimeout

from app.config import Settings
from app.errors import ExtractionError

log = logging.getLogger("app.llm")

# Assignment section 3. Calling anything else disqualifies the submission.
ALLOWED_MODELS = frozenset({
    "google/gemini-3.8-flash",
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-haiku-5.5",
    "anthropic/claude-sonnet-5.5",
    "openai/gpt-6-luna",
    "openai/gpt-6.1-sol",
    "z-ai/glm-5.3-flash",
    "qwen/qwen3-vl-235b-a22b-instruct",
    "deepseek/deepseek-v3.2",
})

# Per million tokens (input, output), from the provider's public listing on 2026-11-07. Used
# only when a gateway returns no cost, so metadata.cost_usd is an estimate in that case.
_PRICES = {
    "google/gemini-3.8-flash": (0.83, 4.12),
    "google/gemini-3.1-pro-preview": (2.00, 12.00),
    "anthropic/claude-haiku-5.5": (0.10, 0.50),
    "anthropic/claude-sonnet-5.5": (2.00, 10.00),
    "openai/gpt-6-luna": (0.10, 0.50),
    "openai/gpt-6.1-sol": (2.00, 10.00),
    "z-ai/glm-5.3-flash": (0.15, 0.50),
    "qwen/qwen3-vl-235b-a22b-instruct": (0.21, 1.90),
    "deepseek/deepseek-v3.2": (0.26, 0.38),
}

_UNUSABLE_REPLY_LIMIT = 3
_MAX_OUTPUT_TOKENS = 32_000
_DB_ERRORS = (psycopg.Error, PoolTimeout, PoolClosed)
_PERMANENT_CODES = {401: "llm_auth_failed", 402: "llm_out_of_credits", 403: "llm_auth_failed"}


class LLMError(ExtractionError):
    """Base class. Messages never include prompts, document text or the key."""


class LLMUnavailable(LLMError):
    def __init__(self, message: str = "The model provider stayed unavailable for the whole retry budget") -> None:
        super().__init__("llm_unavailable", message)


class LLMBadOutput(LLMError):
    def __init__(self, message: str = "The model repeatedly returned unusable output") -> None:
        super().__init__("llm_bad_output", message)


@dataclass
class ModelUsage:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


@dataclass
class Usage:
    """Totals for one job. Every HTTP attempt counts as a call, including failures."""

    models: dict[str, ModelUsage] = field(default_factory=dict)

    def _model(self, model: str) -> ModelUsage:
        return self.models.setdefault(model, ModelUsage())

    def add_call(self, model: str) -> None:
        self._model(model).calls += 1

    def add_reply(self, model: str, usage: object) -> None:
        if not isinstance(usage, dict):
            return
        entry = self._model(model)
        tokens_in = _non_negative(usage.get("prompt_tokens"), int)
        tokens_out = _non_negative(usage.get("completion_tokens"), int)
        entry.tokens_in += tokens_in
        entry.tokens_out += tokens_out
        if isinstance(usage.get("cost"), (int, float)) and not isinstance(usage.get("cost"), bool):
            entry.cost_usd += _non_negative(usage["cost"], float)
        else:
            price_in, price_out = _PRICES.get(model, (0.0, 0.0))
            entry.cost_usd += (tokens_in * price_in + tokens_out * price_out) / 1_000_000

    def as_metadata(self) -> dict:
        items = self.models.values()
        return {
            "model_calls": sum(m.calls for m in items),
            "tokens_in": sum(m.tokens_in for m in items),
            "tokens_out": sum(m.tokens_out for m in items),
            "cost_usd": round(sum(m.cost_usd for m in items), 6),
            "models": {name: {"calls": m.calls, "tokens_in": m.tokens_in, "tokens_out": m.tokens_out,
                              "cost_usd": round(m.cost_usd, 6)} for name, m in self.models.items()},
        }


def _non_negative(value, kind):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value or value < 0:
        return kind(0)
    return kind(value)


@dataclass
class ChatResult:
    content: str
    finish_reason: str | None
    model: str


def parse_retry_after(headers: httpx.Headers, now: float | None = None) -> float | None:
    """Seconds to wait from Retry-After or X-RateLimit-Reset, or None if neither is usable."""
    now = time.time() if now is None else now
    raw = headers.get("retry-after")
    if raw:
        try:
            value = float(raw)
            return max(value, 0.0) if value == value and value != float("inf") else None
        except ValueError:
            pass
        try:
            when = parsedate_to_datetime(raw)
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return max(when.timestamp() - now, 0.0)
        except (TypeError, ValueError):
            pass
    raw = headers.get("x-ratelimit-reset")
    if raw:
        try:
            value = float(raw)
        except ValueError:
            return None
        # OpenRouter sends epoch milliseconds; other gateways send epoch seconds.
        when = value / 1000 if value > 1e11 else value
        if when > 1e9:
            return max(when - now, 0.0)
    return None


class CooldownGate:
    """Makes all workers hold back on a model after any one of them is rate limited.

    Uses the database clock so workers agree on timing. Always tracks locally as well, so
    throttling still works in this process if the database is unavailable.
    """

    def __init__(self, db=None) -> None:
        self._db = db
        self._local: dict[str, float] = {}

    async def _shared_seconds(self, model: str) -> float:
        if self._db is None:
            return 0.0
        try:
            # Short wait: this runs before every model call, and a busy pool must not
            # add seconds to each one. The local cooldown still applies if this fails.
            async with self._db.pool.connection(timeout=0.5) as conn:
                cur = await conn.execute(
                    "SELECT extract(epoch FROM until - now())::double precision AS s FROM llm_cooldowns WHERE model = %s",
                    (model,),
                )
                row = await cur.fetchone()
                return max(row["s"], 0.0) if row else 0.0
        except _DB_ERRORS as exc:
            log.warning("could not read shared cooldown", extra={"error_type": type(exc).__name__})
            return 0.0

    async def seconds_remaining(self, model: str) -> float:
        local = max(self._local.get(model, 0.0) - time.monotonic(), 0.0)
        return max(local, await self._shared_seconds(model))

    async def report(self, model: str, seconds: float) -> None:
        self._local[model] = max(self._local.get(model, 0.0), time.monotonic() + seconds)
        if self._db is None:
            return
        try:
            async with self._db.pool.connection() as conn:
                await conn.execute(
                    "INSERT INTO llm_cooldowns (model, until)"
                    " VALUES (%(m)s, now() + make_interval(secs => %(s)s::double precision))"
                    " ON CONFLICT (model) DO UPDATE SET until = greatest(llm_cooldowns.until, EXCLUDED.until)",
                    {"m": model, "s": seconds},
                )
        except _DB_ERRORS as exc:
            log.warning("could not share cooldown", extra={"error_type": type(exc).__name__})

    async def wait(self, model: str, deadline: float) -> None:
        while True:
            remaining = await self.seconds_remaining(model)
            if remaining <= 0:
                return
            if time.monotonic() + remaining >= deadline:
                raise LLMUnavailable("The model provider's rate limit outlasted the retry budget")
            # Spread callers over a window proportional to the cooldown. If they all woke
            # within the same second they would trip the rate limit again together.
            await asyncio.sleep(remaining + random.uniform(0, min(15.0, 0.5 + remaining * 0.25)))


class _Retry(Exception):
    def __init__(self, reason: str, retry_after: float | None = None, rate_limited: bool = False) -> None:
        super().__init__(reason)
        self.reason, self.retry_after, self.rate_limited = reason, retry_after, rate_limited


class _Unusable(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _parse_error_code(body: dict) -> int | None:
    error = body.get("error")
    code = error.get("code") if isinstance(error, dict) else None
    if isinstance(code, str) and code.isdigit():
        code = int(code)
    return code if isinstance(code, int) and not isinstance(code, bool) else None


def _text_from(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # some gateways return a list of content parts
        return "".join(p.get("text", "") for p in content if isinstance(p, dict) and isinstance(p.get("text"), str))
    return ""


def parse_json_reply(text: str):
    """Parse a reply that may be wrapped in a code fence or surrounded by prose."""

    def reject(name: str):
        raise ValueError(f"{name} is not valid JSON")

    cleaned = re.sub(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$", "", text)
    starts = [i for i in (cleaned.find("{"), cleaned.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON found in reply")
    # strict=False accepts raw tabs and newlines inside strings. Models copy them verbatim from
    # documents (DOCX table cells are tab-separated), and rejecting them fails the whole reply.
    value, _ = json.JSONDecoder(parse_constant=reject, strict=False).raw_decode(cleaned[min(starts):])
    return value


class LLMClient:
    def __init__(self, settings: Settings, gate: CooldownGate | None = None,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._settings = settings
        self._gate = gate or CooldownGate()
        self._semaphore = asyncio.Semaphore(max(settings.llm_max_concurrent, 1))
        self._http = httpx.AsyncClient(transport=transport, follow_redirects=False)

    async def close(self) -> None:
        await self._http.aclose()

    def _backoff(self, attempt: int) -> float:
        s = self._settings
        ceiling = min(s.llm_backoff_max_seconds, s.llm_backoff_base_seconds * 2 ** min(attempt - 1, 20))
        return random.uniform(s.llm_backoff_base_seconds, max(ceiling, s.llm_backoff_base_seconds))

    async def _attempt(self, url: str, payload: dict, model: str, usage: Usage, timeout: float) -> ChatResult:
        headers = {"Authorization": f"Bearer {self._settings.llm_api_key.get_secret_value().strip()}"}
        usage.add_call(model)
        try:
            # Held only for the request itself, so jobs that are backing off don't block others.
            async with self._semaphore:
                response = await self._http.post(url, json=payload, headers=headers, timeout=timeout)
        except httpx.TransportError as exc:  # timeouts, refused and dropped connections
            raise _Retry(type(exc).__name__) from None

        status = response.status_code
        if status == 429:
            raise _Retry("http_429", parse_retry_after(response.headers), rate_limited=True)
        if status in (408, 409, 425) or status >= 500:
            raise _Retry(f"http_{status}", parse_retry_after(response.headers))
        if status in _PERMANENT_CODES:
            raise LLMError(_PERMANENT_CODES[status], f"The model provider refused the request (HTTP {status})")
        if status != 200:
            raise LLMError("llm_rejected", f"The model provider rejected the request (HTTP {status})")

        try:
            body = response.json()
        except ValueError:
            raise _Unusable("invalid_json_body") from None
        if not isinstance(body, dict):
            raise _Unusable("body_not_an_object")
        usage.add_reply(model, body.get("usage"))

        # OpenRouter can report upstream failures inside a 200 response.
        if body.get("error"):
            code = _parse_error_code(body)
            if code == 429:
                raise _Retry("body_429", parse_retry_after(response.headers), rate_limited=True)
            if code in _PERMANENT_CODES:
                raise LLMError(_PERMANENT_CODES[code], f"The model provider refused the request (code {code})")
            if code is not None and 400 <= code < 500 and code not in (408, 409, 425):
                raise LLMError("llm_rejected", f"The model provider rejected the request (code {code})")
            raise _Retry(f"body_error_{code}")

        choices = body.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
        message = choice.get("message") if choice else None
        content = _text_from(message.get("content")) if isinstance(message, dict) else ""
        if not content.strip():
            raise _Unusable("empty_reply")
        return ChatResult(content=content, finish_reason=choice.get("finish_reason"), model=model)

    async def chat(self, model: str, messages: list[dict], usage: Usage, *,
                   budget_seconds: float | None = None, temperature: float = 0.0,
                   max_tokens: int | None = None, response_format: dict | None = None) -> ChatResult:
        s = self._settings
        if model not in ALLOWED_MODELS:
            raise LLMError("llm_model_not_allowed", f"Model {model!r} is not on the allowed list")
        if not s.llm_configured:
            raise LLMError("llm_not_configured", "LLM_BASE_URL or LLM_API_KEY is not set")

        url = s.llm_base_url.strip().rstrip("/") + "/chat/completions"
        payload: dict = {"model": model, "messages": messages, "temperature": temperature}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format

        budget = s.llm_call_budget_seconds if budget_seconds is None else budget_seconds
        deadline = time.monotonic() + budget
        attempt = unusable = 0
        while True:
            await self._gate.wait(model, deadline)
            attempt += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LLMUnavailable()
            timeout = min(s.llm_request_timeout_seconds, remaining)
            try:
                return await self._attempt(url, payload, model, usage, timeout)
            except _Retry as retry:
                delay = max(self._backoff(attempt), retry.retry_after or 0.0)
                reason, rate_limited = retry.reason, retry.rate_limited
            except _Unusable as bad:
                unusable += 1
                reason, rate_limited, delay = bad.reason, False, self._backoff(attempt)
                if unusable >= _UNUSABLE_REPLY_LIMIT:
                    log.error("model kept returning unusable replies", extra={"model": model, "reason": reason})
                    raise LLMBadOutput() from None

            log.warning("model call will be retried", extra={
                "model": model, "attempt": attempt, "reason": reason, "delay_seconds": round(delay, 2)})
            if time.monotonic() + delay >= deadline:
                raise LLMUnavailable()
            if rate_limited:
                await self._gate.report(model, delay)  # the wait at the top of the loop sleeps
            else:
                await asyncio.sleep(delay)

    async def chat_json(self, model: str, messages: list[dict], usage: Usage, *, parse_attempts: int = 3, **kwargs):
        """Like chat(), but returns parsed JSON and retries replies that can't be parsed."""
        last = "no reply"
        grown = False
        for _ in range(max(parse_attempts, 1)):
            result = await self.chat(model, messages, usage, **kwargs)
            if result.finish_reason == "length":
                # Cut off mid-reply: even if it parses, it's incomplete. The same request
                # would be cut off the same way, so retry once with more room, then stop.
                last = "reply was cut off at the token limit"
                current = kwargs.get("max_tokens")
                if grown or (current or 0) >= _MAX_OUTPUT_TOKENS:
                    break
                kwargs = {**kwargs, "max_tokens": min(current * 2, _MAX_OUTPUT_TOKENS) if current else _MAX_OUTPUT_TOKENS}
                grown = True
                continue
            try:
                return parse_json_reply(result.content)
            except (ValueError, RecursionError) as exc:
                last = type(exc).__name__
        log.error("model kept returning invalid JSON", extra={"model": model, "reason": last})
        raise LLMBadOutput()
