# Temporal validity / as-of filtering

`atomzero.temporal` treats an explicit validity window as a first-class signal.
A hit may carry `valid_from` and/or `valid_until`; the module decides whether it
is valid for an as-of date, or why it is excluded. It is deterministic, local
and makes no model calls.

```powershell
$env:PYTHONPATH="$PWD\src"
python -m atomzero.temporal --file hits.json --as-of 2026-11-23 --json
python -m atomzero.temporal --file hits.json --exclude-undated
```

```python
from atomzero.temporal import filter_as_of

report = filter_as_of(hits, as_of="2026-11-23")
# {"included": [...], "excluded": [...], "summary": {...},
#  "weights_changed": False, "method": "deterministic_as_of_validity_v1"}
```

## Status

| Status | Meaning |
|---|---|
| `valid` | `as_of` is inside the inclusive window (or only the available bound excludes nothing) |
| `expired` | `as_of > valid_until` |
| `not_yet` | `as_of < valid_from` |
| `undated` | neither `valid_from` nor `valid_until` is present |

`days_remaining` is `valid_until - as_of`; `days_until_start` is
`valid_from - as_of`. Both can be negative in their non-valid states.

## Filtering

- `keep_undated=True` (default) keeps undated evidence. Set it false for a
  strict as-of view.
- The report keeps both included and excluded entries, so a caller can show
  why evidence disappeared.
- `as_of` defaults to local today and accepts a `date`, `datetime` or ISO
  string.
- Invalid dates, non-list hits and non-object entries raise `ValueError`.

## HTTP

`POST /v1/temporal/filter` accepts `hits` (maximum 50), optional `as_of` and
`keep_undated`. Only source/title/locator/chunk fields plus `valid_from` and
`valid_until` are passed to the filter. It is stateless.

```powershell
$body = @{
  as_of = "2026-11-23"
  hits = @(
    @{ source = "policy-v1"; valid_until = "2025-12-31" },
    @{ source = "policy-v2"; valid_from = "2026-01-01" }
  )
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/temporal/filter `
  -ContentType "application/json" -Body $body
```

## Honest limits

- The window is only as good as the metadata. The module does not infer
  validity from prose, file mtime or web freshness.
- An `undated` hit is not automatically valid or invalid; the caller chooses
  whether to keep it.
- Inclusive boundaries are a convention; an event that expires during the
  as-of day needs a timestamp, not a date.
- This filters evidence; it does not prove that the selected version is
  correct or that the excluded one is false.

## Verification

```powershell
$env:PYTHONPATH="$PWD\src"
python -m pytest tests/test_temporal.py tests/test_chat_server_docs.py::test_http_temporal_filter -q
python -m aisc check docs/aisc/atomzero-temporal.toml --strict
```

The AISC spec declares three verified components, no irreversible steps and
`min_assurance = 0.90`.
