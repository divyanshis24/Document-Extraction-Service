#!/usr/bin/env python3
"""Drive the running service the way the scoring harness does, and measure it.

  python scripts/load.py samples            # the sample set through the service -> outputs/
  python scripts/load.py synthetic -n 300   # 300 generated documents, stub extractor, no model calls
  python scripts/load.py synthetic -n 300 --kill-worker 20   # kill one worker 20s in; nothing may be lost

Keeps up to --in-flight jobs submitted and not yet terminal, polls each every --poll seconds, and
reports: completion rate, p50/p95/max latency from submission to terminal status, the slowest
POST (the contract allows 2s), and whether any job was lost (never terminal) or duplicated.

`samples` needs the real extractor and a key in .env (about $0.45 for the eight documents).
`synthetic` is for load, latency and crash recovery: run the workers with EXTRACTOR=stub.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import validate  # noqa: E402

TERMINAL = {"SUCCESS", "PARTIAL", "FAILED"}
SYNTHETIC_FIELDS = {"name": "synthetic", "fields": {
    "unit": {"description": "Unit identifier.", "primary": True},
    "rent": {"description": "Amount in dollars."},
    "start": {"description": "Start date, YYYY-MM-DD."},
}}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))]


class Run:
    def __init__(self, base_url: str, in_flight: int, poll: float, timeout: float, org: str | None) -> None:
        self.base_url, self.in_flight, self.poll, self.timeout = base_url.rstrip("/"), in_flight, poll, timeout
        self.headers = {"X-Organization-Id": org} if org else {}
        self.records: list[dict] = []
        self.post_latencies: list[float] = []
        self.slots = asyncio.Semaphore(in_flight)

    async def submit(self, client: httpx.AsyncClient, name: str, content: bytes, fields: dict) -> dict:
        started = time.monotonic()
        response = await client.post(f"{self.base_url}/jobs", headers=self.headers,
                                     files={"file": (name, content)}, data={"schema": json.dumps(fields)})
        self.post_latencies.append(time.monotonic() - started)
        if response.status_code != 202:
            raise RuntimeError(f"POST /jobs for {name}: HTTP {response.status_code} {response.text[:200]}")
        return response.json()

    async def wait_terminal(self, client: httpx.AsyncClient, job_id: str) -> dict | None:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            await asyncio.sleep(self.poll)
            response = await client.get(f"{self.base_url}/jobs/{job_id}", headers=self.headers)
            if response.status_code == 200 and response.json()["status"] in TERMINAL:
                return response.json()
        return None

    async def one(self, client: httpx.AsyncClient, name: str, content: bytes, fields: dict, on_done=None) -> None:
        async with self.slots:
            submitted = time.monotonic()
            job = await self.submit(client, name, content, fields)
            final = await self.wait_terminal(client, job["job_id"])
            record = {"name": name, "job_id": job["job_id"], "status": final["status"] if final else "LOST",
                      "latency": time.monotonic() - submitted, "final": final}
            self.records.append(record)
            if on_done:
                on_done(record)

    def report(self) -> bool:
        statuses = {s: sum(1 for r in self.records if r["status"] == s) for s in ("SUCCESS", "PARTIAL", "FAILED", "LOST")}
        done = [r["latency"] for r in self.records if r["status"] in TERMINAL]
        ids = [r["job_id"] for r in self.records]
        duplicates = len(ids) - len(set(ids))
        total = len(self.records)
        completed = statuses["SUCCESS"] + statuses["PARTIAL"]
        print(f"\njobs {total}: {statuses}")
        print(f"completion {completed}/{total} = {completed / max(total, 1):.1%}  (terminal incl. FAILED: {(total - statuses['LOST']) / max(total, 1):.1%})")
        if done:
            print(f"latency to terminal  p50 {percentile(done, 50):.1f}s  p95 {percentile(done, 95):.1f}s  "
                  f"max {max(done):.1f}s  mean {statistics.mean(done):.1f}s")
        print(f"POST /jobs latency   p50 {percentile(self.post_latencies, 50) * 1000:.0f}ms  "
              f"max {max(self.post_latencies, default=0) * 1000:.0f}ms  (limit 2000ms)")
        print(f"duplicated job ids: {duplicates}   lost (never terminal): {statuses['LOST']}")
        ok = statuses["LOST"] == 0 and duplicates == 0 and max(self.post_latencies, default=0) < 2.0
        return ok


async def run_samples(args) -> int:
    rows = list(csv.DictReader((ROOT / "sample" / "manifest.csv").open()))
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    run = Run(args.base_url, args.in_flight, args.poll, args.timeout, args.org)

    def save(record: dict) -> None:
        final = record["final"] or {}
        row = next(r for r in rows if r["file"] == record["name"])
        print(f"{row['document_id']:14s} {record['status']:8s} {record['latency']:6.1f}s", flush=True)
        if final.get("result"):
            result = {**final["result"], "schema": row["schema_name"]}
            (out_dir / f"{row['document_id']}.json").write_text(json.dumps(result, indent=1))

    async with httpx.AsyncClient(timeout=30) as client:
        await asyncio.gather(*[
            run.one(client, row["file"], (ROOT / "sample" / "sample-data-documents" / row["file"]).read_bytes(),
                    json.loads((ROOT / "sample" / "schemas" / f"{row['schema_name']}.json").read_text()), save)
            for row in rows])
    ok = run.report()
    print()
    code = subprocess.call([sys.executable, str(ROOT / "tools" / "validate.py"), str(out_dir), "--schema-dir",
                            str(ROOT / "sample" / "schemas"), "--truth-dir", str(ROOT / "sample" / "sample-data-output-validated")])
    return 0 if ok and code == 0 else 1


RUN_ID = uuid.uuid4().hex[:8]


def synthetic_document(index: int) -> bytes:
    # Unique per run: an identical document and field list would be deduplicated to the
    # job from the last run, and nothing would be measured.
    lines = [f"# run {RUN_ID}", "unit,rent,start"] + [f"U-{index}-{i},{1000 + i},2024-01-{(i % 28) + 1:02d}" for i in range(50)]
    return "\n".join(lines).encode()


async def run_synthetic(args) -> int:
    run = Run(args.base_url, args.in_flight, args.poll, args.timeout, args.org)
    started = time.monotonic()
    killer = None
    if args.kill_worker:
        async def kill_later() -> None:
            await asyncio.sleep(args.kill_worker)
            names = subprocess.run(["docker", "compose", "ps", "-q", "worker"], capture_output=True, text=True, cwd=ROOT).stdout.split()
            if names:
                # SIGKILL, so nothing is released gracefully, then start it again as a supervisor
                # would. Jobs it held stay RUNNING until their lease expires, then get reclaimed.
                print(f"--- crashing worker {names[0][:12]} at {time.monotonic() - started:.0f}s", flush=True)
                subprocess.run(["docker", "kill", names[0]], capture_output=True, cwd=ROOT)
                await asyncio.sleep(2)
                subprocess.run(["docker", "start", names[0]], capture_output=True, cwd=ROOT)
        killer = asyncio.create_task(kill_later())

    def progress(record: dict) -> None:
        n = len(run.records)
        if n % 25 == 0 or record["status"] in ("FAILED", "LOST"):
            print(f"{n:4d} done  {time.monotonic() - started:5.0f}s  last: {record['status']} {record['latency']:.1f}s", flush=True)

    async with httpx.AsyncClient(timeout=30) as client:
        # Same document and field list twice must not do the work twice.
        first = await run.submit(client, "dup.csv", synthetic_document(-1), SYNTHETIC_FIELDS)
        second = await run.submit(client, "dup.csv", synthetic_document(-1), SYNTHETIC_FIELDS)
        print(f"idempotent resubmission: {'same job' if first['job_id'] == second['job_id'] else 'DIFFERENT JOBS'}")
        await asyncio.gather(*[run.one(client, f"doc-{i}.csv", synthetic_document(i), SYNTHETIC_FIELDS, progress)
                               for i in range(args.jobs)])
    if killer:
        killer.cancel()
    elapsed = time.monotonic() - started
    ok = run.report() and first["job_id"] == second["job_id"]
    print(f"throughput {len(run.records) / elapsed:.1f} jobs/s over {elapsed:.0f}s with {args.in_flight} in flight")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["samples", "synthetic"])
    ap.add_argument("--base-url", default="http://localhost:8080")
    ap.add_argument("--in-flight", type=int, default=10, help="documents kept in flight, like the harness")
    ap.add_argument("--poll", type=float, default=2.0)
    ap.add_argument("--timeout", type=float, default=1800.0, help="seconds a job may take to reach a terminal status")
    ap.add_argument("--org", help="X-Organization-Id header")
    ap.add_argument("--out", default="outputs", help="samples: where result JSON goes")
    ap.add_argument("-n", "--jobs", type=int, default=200, help="synthetic: how many documents")
    ap.add_argument("--kill-worker", type=float, metavar="SECONDS", help="synthetic: kill one worker this many seconds in")
    args = ap.parse_args()
    return asyncio.run(run_samples(args) if args.mode == "samples" else run_synthetic(args))


if __name__ == "__main__":
    sys.exit(main())
