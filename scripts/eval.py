#!/usr/bin/env python3
"""Score the extractor on the sample set, with real money protected.

  python scripts/eval.py --docs boxscore_001                 # one document
  python scripts/eval.py --kind xlsx --model-fast openai/gpt-6-luna
  python scripts/eval.py                                     # everything

- Replies are cached on disk, so re-running after a scoring or code change costs nothing
  unless the prompts changed. Use --no-cache to force fresh calls.
- A ledger tracks real spending across runs, and a run stops before passing --max-total.
- The key comes from LLM_API_KEY or .env and is never printed.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import validate  # noqa: E402

from app.config import Settings  # noqa: E402
from app.extraction.extractor import RealExtractor  # noqa: E402
from app.jobs import ClaimedJob  # noqa: E402
from app.llm import LLMClient  # noqa: E402
from app.logging_config import configure_logging  # noqa: E402
from app.result import build_result  # noqa: E402

STATE = ROOT / ".eval"
LEDGER = STATE / "ledger.json"


class CachingTransport(httpx.AsyncBaseTransport):
    """Replays identical requests from disk, and refuses to spend past the limit."""

    def __init__(self, use_cache: bool, max_total: float) -> None:
        self._real = httpx.AsyncHTTPTransport()
        self.use_cache, self.max_total = use_cache, max_total
        self.dir = STATE / "cache"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.spent = json.loads(LEDGER.read_text())["spent_usd"] if LEDGER.exists() else 0.0
        self.hits = self.misses = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        path = self.dir / f"{hashlib.sha256(body).hexdigest()}.json"
        if self.use_cache and path.exists():
            self.hits += 1
            return httpx.Response(200, content=path.read_bytes(), headers={"content-type": "application/json"})
        if self.spent >= self.max_total:
            raise SystemExit(f"Stopped: ledger shows ${self.spent:.4f}, at the --max-total limit of ${self.max_total:.2f}")
        self.misses += 1
        response = await self._real.handle_async_request(request)
        content = await response.aread()
        if response.status_code == 200:
            try:
                usage = json.loads(content).get("usage") or {}
                cost = usage.get("cost")
                if isinstance(cost, (int, float)):
                    self.spent += cost
                    LEDGER.write_text(json.dumps({"spent_usd": round(self.spent, 6)}))
                path.write_bytes(content)
            except ValueError:
                pass
        # aread() already decompressed the body, so drop the headers that describe the wire
        # encoding, or the client would try to decompress it a second time.
        headers = [(k, v) for k, v in response.headers.multi_items()
                   if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")]
        return httpx.Response(response.status_code, content=content, headers=headers)


async def run_document(row: dict, settings: Settings, transport: CachingTransport, out_dir: Path) -> dict:
    documents = ROOT / "sample" / "sample-data-documents"
    schema = json.loads((ROOT / "sample" / "schemas" / f"{row['schema_name']}.json").read_text())
    content = (documents / row["file"]).read_bytes()
    client = LLMClient(settings, transport=transport)
    job = ClaimedJob(id=uuid.uuid4(), lease_token=uuid.uuid4(), file_name=row["file"], file_type=row["kind"],
                     field_list=schema, document_sha256="", attempts=1, elapsed_seconds=0, remaining_seconds=1700)

    async def progress(stage: str) -> None:
        pass

    started = time.monotonic()
    try:
        output = await RealExtractor(client, settings).extract(job, content, progress)
    finally:
        await client.close()
    latency = int((time.monotonic() - started) * 1000)
    result = build_result(output.entries, list(schema["fields"]), output.metadata, latency)
    result["schema"] = row["schema_name"]
    path = out_dir / f"{row['document_id']}.json"
    path.write_text(json.dumps(result, indent=1))

    errors = validate.validate_contract(result, schema)
    truth = json.loads((ROOT / "sample" / "sample-data-output-validated" / f"{row['document_id']}.json").read_text())
    return {"id": row["document_id"], "errors": errors, "report": validate.score(result, truth, schema),
            "meta": result["metadata"], "partial": output.partial}


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", nargs="*", help="document ids")
    ap.add_argument("--kind", help="pdf, docx, xlsx or csv")
    ap.add_argument("--max-total", type=float, default=7.0, help="stop before the ledger passes this many dollars")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default=str(STATE / "outputs"))
    ap.add_argument("--details", action="store_true", help="show the fields that scored below 1")
    for name in ("fast", "strong", "table", "vision"):
        ap.add_argument(f"--model-{name}")
    args = ap.parse_args()

    env = dotenv_values(ROOT / ".env")
    overrides = {f"llm_model_{n}": getattr(args, f"model_{n}") for n in ("fast", "strong", "table", "vision") if getattr(args, f"model_{n}")}
    settings = Settings(llm_base_url=env.get("LLM_BASE_URL") or Settings().llm_base_url,
                        llm_api_key=env.get("LLM_API_KEY") or Settings().llm_api_key, log_level="WARNING",
                        **overrides)
    configure_logging("WARNING")
    if not settings.llm_configured:
        raise SystemExit("LLM_BASE_URL and LLM_API_KEY must be set, in the environment or .env")

    STATE.mkdir(exist_ok=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((ROOT / "sample" / "manifest.csv").open()))
    rows = [r for r in rows if (not args.docs or r["document_id"] in args.docs) and (not args.kind or r["kind"] == args.kind)]
    transport = CachingTransport(not args.no_cache, args.max_total)
    before = transport.spent

    scores = []
    for row in rows:
        started = time.monotonic()
        try:
            r = await run_document(row, settings, transport, out_dir)
        except Exception as exc:  # noqa: BLE001 - report it and carry on with other documents
            print(f"FAILED  {row['document_id']}: {type(exc).__name__}: {str(exc)[:200]}")
            scores.append(0.0)
            continue
        rep = r["report"]
        scores.append(rep["score"])
        print(f"{r['id']:14s} score {rep['score']:.3f}  rows {rep['rows']['predicted']}/{rep['rows']['truth']}  "
              f"halluc {len(rep['hallucinations'])}  miss {len(rep['misses'])}  ${r['meta']['cost_usd']:.4f}  "
              f"calls {r['meta']['model_calls']}  {time.monotonic() - started:.0f}s"
              f"{'  PARTIAL' if r['partial'] else ''}{'  INVALID' if r['errors'] else ''}")
        if args.details:
            for key, value in sorted(rep["per_field"].items(), key=lambda kv: kv[1]):
                if value < 1:
                    print(f"    {key:38s} {value:.2f}")
            if rep["extra_rows"]:
                print(f"    extra rows: {rep['extra_rows'][:8]}")
            if rep["missing_rows"]:
                print(f"    missing rows: {rep['missing_rows'][:8]}")
    if scores:
        print(f"\nMEAN {sum(scores) / len(scores):.3f} over {len(scores)} documents")
    print(f"this run spent ${transport.spent - before:.4f} ({transport.misses} calls, {transport.hits} from cache); "
          f"ledger total ${transport.spent:.4f} of ${args.max_total:.2f} limit")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
