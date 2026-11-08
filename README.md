# Document Extraction Service

**Role: Software Developer.**

A service that takes a document (PDF, DOCX, XLSX or CSV) and a field list, and returns the
extracted values as JSON with a citation and a confidence for each one. Jobs are queued,
worked in the background by any number of workers, and never lost or duplicated.

## Run it

```bash
cp .env.example .env        # put the OpenRouter key in LLM_API_KEY
docker compose up --build   # API on :8080, Postgres, two workers
curl localhost:8080/health  # {"status":"ok"} within a few seconds
```

Submit a document and poll for the result:

```bash
curl -F file=@document.pdf -F schema=@fields.json localhost:8080/jobs
#  -> {"job_id": "...", "status": "PENDING", ...}
curl localhost:8080/jobs/<job_id>
#  -> status PENDING | RUNNING | SUCCESS | PARTIAL | FAILED, with "result" once terminal
```

Optional: `X-Organization-Id` header (tenants never see each other's jobs) and a
`webhook_url` form field (called once the job is terminal, with retries).

More throughput is more workers: `docker compose up --scale worker=4`. Each worker runs
`WORKER_CONCURRENCY` (default 15) documents at a time; they spend nearly all their time waiting
on the model, so the limit is the provider's rate, not CPU.

## Check it

```bash
make test       # unit tests, no database, no network, ~10s
make test-db    # the same plus the Postgres-backed queue, worker and API tests (throwaway container)
make up         # build, start, wait for /health
make samples    # the sample set through the running service -> outputs/, then the validator
make load       # 200 synthetic jobs, 10 in flight, one worker crashed mid-run; stub extractor, no model calls
make eval       # accuracy on the sample set, cached and spend-capped (needs LLM_API_KEY in .env)
make check      # all of the above except eval
```

## Scores on the sample set

`outputs/` holds the service's result for every sample document, produced by `make samples`
(all eight submitted at once, as the harness does).

```
$ python3 tools/validate.py outputs/ --schema-dir sample/schemas --truth-dir sample/sample-data-output-validated
SCORE    boxscore_001.json: 1.000   rows 10/10     hallucinations=0 misses=0
SCORE    budget_001.json:   0.999   rows 55/55     hallucinations=0 misses=0
SCORE    lease_001.json:    0.900   rows 1/1       hallucinations=0 misses=0
SCORE    lease_002.json:    0.850   rows 1/1       hallucinations=0 misses=0
SCORE    lease_003.json:    0.850   rows 1/1       hallucinations=0 misses=0
SCORE    om_001.json:       1.000   rows 1/1       hallucinations=0 misses=0
SCORE    rentroll_001.json: 1.000   rows 325/325   hallucinations=0 misses=0
SCORE    t12_001.json:      0.997   rows 119/119   hallucinations=0 misses=0

MEAN SCORE over 8 scored documents: 0.949
```

The validator accepts every file (exit 0, no `INVALID` line). There are no hallucinated values anywhere: a value the pipeline cannot
find in the document becomes null. The lease scores are wording differences on free-text fields
("base year 1993" vs "base year 1993 for direct expenses and taxes"), which the validator's exact
match counts as misses.

| Measured on the same run | |
|---|---|
| Cost | $0.055 per document on average ($0.016 for a box score, $0.13 for a 21-page scanned lease) |
| Latency, submission to terminal | p50 34 s, p95 74 s, max 74 s |
| `POST /jobs` | p50 66 ms, max 156 ms (contract: under 2 s) |
| Load (`make load`) | 150 jobs with one worker killed mid-run: 150/150 completed, 0 lost, 0 duplicated; the killed worker's jobs were reclaimed after their 60 s lease |

## Layout

```
app/main.py            FastAPI: POST /jobs, GET /jobs/{id}, GET /health
app/jobs.py            the job store: atomic claim, leases, fenced writes, dedupe, sweep
app/worker.py          worker loop: heartbeat, deadline, crash and shutdown handling
app/llm.py             the model client: retries, backoff, shared rate-limit cooldown, cost
app/webhooks.py        at-least-once webhook delivery
app/extraction/        the pipeline: parsing, field typing, grounding, spreadsheet plans
contract/, tools/      the fixed interface and the validator, as supplied
scripts/eval.py        accuracy on the sample set, cached and spend-capped
scripts/load.py        drives the running service: the sample set, or synthetic load with a crash
tests/                 453 tests; the database-backed ones run when TEST_DATABASE_URL is set
```

See [DESIGN.md](DESIGN.md) for the decisions and what to watch when running it.
