# Interface contract

Everything in this folder is fixed. Our evaluation harness is written against it, so a
submission that deviates cannot be scored.

| File | What it is |
|---|---|
| `openapi.yaml` | The three HTTP endpoints your service must expose: `POST /jobs`, `GET /jobs/{job_id}`, `GET /health`. |
| `output.schema.json` | JSON Schema for the result object returned under `result` once a job is `SUCCESS` or `PARTIAL`. |
| `schema.example.json` | What a field list looks like. You get one per document type in the sample set. |
| `output.example.json` | A complete, valid result for the example above. |
| `../tools/validate.py` | Run it on your own output. If it prints `VALID`, our harness can score you. With `--truth` it also scores you against the sample answers: a fixed match after normalisation (numbers within 0.5%, dates in any common format, case and whitespace ignored, currency symbols stripped), not a semantic one. Our scoring is slightly more lenient on free-text fields, so a cell the validator marks correct is correct for us too. |

## Runtime rules

1. `docker compose up` from your repo root starts everything. No manual steps.
2. The service listens on port **8080** and `GET /health` returns `200 {"status":"ok"}` within 60 seconds.
3. **All LLM calls go through two environment variables**, read at runtime:

   ```
   LLM_BASE_URL   OpenAI-compatible base URL. During the assignment: https://openrouter.ai/api/v1
   LLM_API_KEY    the key to send as `Authorization: Bearer <key>`. During the assignment: the key we emailed you
   ```

   During scoring we set both to our own gateway and our own key. Hardcoded keys, other
   provider endpoints, or calls that bypass these variables mean the submission cannot be scored.
4. Model ids are OpenRouter ids. Use only models from the list in the assignment brief (section 3).
5. `POST /jobs` returns within 2 seconds. Extraction happens in the background.
6. Our harness polls `GET /jobs/{job_id}` every 2 seconds and allows **1800 seconds** (30 minutes) per
   document to reach a terminal status (`SUCCESS`, `PARTIAL`, `FAILED`), counted from submission, queue time included. Past that, the
   document scores zero.
7. Terminal statuses never change.

## What you receive per document type

A JSON file with a `name` and a `fields` map. Each field has a display `name` and a
`description`. That is all the guidance there is: whether a field is a number, a date, a
percentage, or occasionally a list, is for you to infer from the description. The descriptions
are ours, not sacred: rewrite them, add examples, add hints, split or merge them internally. The
only thing that must survive is the field keys in your output.

Your service must accept PDF, DOCX, XLSX and CSV through the same `POST /jobs`, with any field
list that follows this shape, including ones not in the sample set. Refusing a supported type
with a 400, or failing because a field list is unfamiliar, counts as a failed document.

Some documents are tables (a rent roll has one row per unit, a box score one row per unit
type). In those field lists exactly one field is marked `"primary": true`. It is the column
that identifies a row. We use it to match your rows to ours when scoring, whatever order you
return them in. You do not have to do anything with it.

## How to shape the output

The result is a list of **entries**, one per row of the document, plus a **metadata** block.

- A document with no table (a lease, a loan summary) has exactly **one** entry.
- A document with a table has **one entry per row**.
- **Every entry carries every field in the field list.** Fields that apply to the whole
  document (property name, report date) repeat the same value in every entry.
- Each field is an object: `{"value": ..., "citation": "...", "confidence": 0.0-1.0}`.
  - `value`: a JSON number for numeric fields (no strings, no `$`, no thousands separators),
    `YYYY-MM-DD` for dates, a string otherwise. Percentages exactly as the document prints
    them. `null` when the value is genuinely not in the document. **Never guess**: we count
    hallucinated values separately, and they cost more than a null.
  - `citation`: a short verbatim snippet with where it came from, e.g.
    `"[Page 3] Base Rent: $4,250.00 per month"` or `"[Sheet1!F4] 4250"`. Empty string when value is null.
  - `confidence`: your own estimate that the value is correct.
  - Optional bonus: `page_numbers` (PDFs) and `source_locations` (cell references, XLSX/CSV).
- `metadata` reports what your service did for that document: `model_calls`, `tokens_in`,
  `tokens_out`, `cost_usd`, `latency_ms`, optionally a per-model breakdown. We measure cost
  independently through our gateway, so this is for your tracking and for us to understand
  your pipeline, not a number you can game.

See `output.example.json` for a two-row box score, including a field returned as null.
