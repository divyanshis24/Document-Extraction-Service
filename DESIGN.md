# Design

For whoever runs this on Monday. Role: Software Developer.

## The job model

`POST /jobs` stores the document and field list in Postgres and returns; nothing else happens on
the request path. Workers claim jobs with one atomic `UPDATE … WHERE id = (SELECT … FOR UPDATE
SKIP LOCKED)`, so any number of workers can run and each job is claimed by exactly one. A claim
is a 60 s lease, renewed by a heartbeat; every write a worker makes is fenced on its lease token,
so a worker that lost its lease can't overwrite the next owner's work.

- **A crashed worker loses nothing.** Its jobs stay `RUNNING` until the lease expires, then another
  worker takes them with `attempts + 1`. `make load` kills a worker mid-run to prove it.
- **The same document and field list is one job.** A partial unique index on (org, document
  hash, field-list hash) for non-failed jobs makes resubmission return the existing job, even
  under concurrent submissions.
- **Terminal statuses never change.** A database trigger refuses any update to a terminal row.
- **Nothing stays pending forever.** The worker cancels at 1740 s from submission; a sweeper in
  every worker *and* in the API process fails anything past the deadline or past three attempts.
- **Shutdown is graceful.** SIGTERM releases in-flight jobs without using an attempt.
- **Tenants are separate.** `X-Organization-Id` is part of every lookup.

Postgres is the queue on purpose: one durable store, transactional with the job rows, nothing
else to run. A broker would add an operational dependency for no gain at this scale.

## The provider will throttle and fail

One client retries 429s and 5xx with backoff and honours `Retry-After`. A rate limit seen by one
worker goes into a shared cooldown table so all workers back off, and they wake spread over a
window rather than together. Each call gets only the time left before the job's deadline, so a
slow provider ends a job `PARTIAL` with what it has, not `FAILED` with nothing. A reply cut off
at the token limit is retried once with more room; unusable replies fail the call and the
pipeline degrades (first pass kept, recheck skipped) rather than failing the document.

## Extraction

| Document | Approach | Model |
|---|---|---|
| Scanned page | transcribed from the image, then treated as text | Haiku |
| Spreadsheet table | the model writes a *plan* once from sampled rows; code applies it to every row | Sonnet plans, Haiku fills the few judgement fields |
| Table on pages | read page by page, previous page as context | Haiku |
| Single-entry document | one cheap read; a stronger model only for what failed verification or needs computing | Haiku, then Sonnet |

The plan is why a 325-unit rent roll costs $0.04: three calls, not 325, and values are copied
from cells exactly with the cell as the citation. **Nothing the model says is trusted until code
finds it in the document**: numbers as whole numbers (250 is not inside 1,250), dates in any
printed form, text after normalisation. What isn't there becomes null. That is what keeps
hallucinations at zero, and it is tested with scripted model replies that lie.

Any field list works: types come from explicit wording ("as a number", "YYYY-MM-DD"), else from
one cached model reading of the list, else generic keyword rules. No field key is known to the
code. Rewording every rent roll description still scored 1.000.

## Budget

About $3.60 of $10: $2.25 on the eval loop (`scripts/eval.py` caches every reply and refuses to
spend past a cap, so re-scoring after a code change is free) and three full runs through the
service at about $0.45 each. A document costs $0.016 (box score) to $0.13 (21 scanned pages).

## What broke

- **The first rent roll run hallucinated five values.** The last unit's block swallowed the
  summary tables below it; the sample check saw a $916,396 rent, rejected three fields, and a
  cheap re-read invented values. Now a block ends at any heading or total row, and a check
  objection stands only if the planned value isn't in the row's cells or the check names one
  that is. A cheap model's opinion can't discard what code copied from the sheet.
- **A fresh plan dropped 22 subtotal rows** the answers count as rows; the planner is now told a
  row with its own identifier is a row.
- **Idempotency fooled the load test**: identical synthetic documents were deduplicated to the
  previous run and a run "finished" in two seconds doing nothing. Each run is now salted.
- **`docker kill` is not a crash**: the restart policy ignores it, and PID 1 drops a SIGKILL
  from inside its own container. The crash test kills and restarts the container explicitly.
- **A tab copied from a DOCX table broke strict JSON parsing** three times over; parsing is
  now lenient about control characters inside strings.
- **Twenty concurrent 30 MB uploads exhausted the API's five connections** and returned 503.
  Pool of 20 now, plus an index for the one query that scanned the whole table.

## Tradeoffs, and another week

Free-text lease fields score 0.85–0.90 locally because the validator wants exact wording; I spent
the time on nulls, numbers and rows, where errors cost most. Two workers of 15 slots cover the
harness's ten in flight with room; more is `--scale worker=N`, up to about ten before Postgres
`max_connections` needs raising. Memory limits and lazy page rendering bound what one huge
document can do.

With another week: chunk very long documents instead of truncating at 150k tokens; a soft
deadline that returns the fields extracted so far instead of cancelling the whole document;
metrics (queue depth, p95 per stage, cost) exported rather than only in the logs; and per-field
model routing learned from the eval loop instead of decided by verification failure at run time.
