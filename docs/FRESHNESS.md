# Freshness algebra

`atomzero.freshness` scores local retrieval hits by the calendar dates written
in their text. It is deterministic, offline and makes no model calls. It does
not change retrieval weights.

```python
from atomzero.freshness import assess_hit, audit_freshness

item = assess_hit(
    {"source": "note.md", "title": "Note",
     "chunk": "Updated 2025-12-01."},
    now="2026-01-01",
)
# {"source": "note.md", "title": "Note", "date": "2025-12-01",
#  "age_days": 31, "status": "fresh", "span": [8, 18],
#  "precision": "day"}

report = audit_freshness([hit], now="2026-01-01")
# {"hits": [...], "summary": {...}, "weights_changed": False,
#  "method": "deterministic_source_freshness_v1"}
```

## API

- `assess_hit(hit, now=None, max_age_days=365, fresh_days=90)` assesses one
  mapping. `now` is a `date`, `datetime` or ISO date/datetime string; the
  default is the local calendar date.
- `audit_freshness(hits, now=None, max_age_days=365, fresh_days=90)` returns
  `{hits, summary, weights_changed: False, method}`.
- `summary` contains `hit_count`, `status_counts`, `source_count`,
  `oldest_date`, `newest_date`, `stale_sources` and `undated_sources`.
- `main(argv=None)` is the CLI entry point.

Dates are found with `atomzero.contradictions.date_matches` in `chunk`,
`snippet` and `title` text. The latest date not later than `now` wins. When
several dates are in the future, the nearest one is reported. `span` is the
`[start, end]` offset of the chosen date inside the field where it was found;
`precision` is the matcher precision (`day` today). Invalid calendar dates such
as `2026-02-31` are skipped.

`source` falls back to `title`, then to `hit-N`. `title` falls back to
`source`, then to an empty string.

## Statuses

| Status | Meaning |
| --- | --- |
| `fresh` | `age_days <= fresh_days` |
| `aging` | `fresh_days < age_days <= max_age_days` |
| `stale` | `age_days > max_age_days` |
| `future` | no past date; the nearest future date is used and `age_days` is negative |
| `undated` | no valid date was found in the hit text |

Thresholds must satisfy `max_age_days >= fresh_days >= 0`; otherwise a
`ValueError` is raised. Non-list `hits`, non-mapping hit entries and invalid
`now` values are also rejected with `ValueError`.

## CLI

```powershell
$env:PYTHONPATH="$PWD\src"
python -m atomzero.freshness --file hits.json --json
python -m atomzero.freshness --file hits.json --now 2026-01-01 --fresh-days 90
python -m atomzero.freshness --max-age-days 365 --fresh-days 90
```

Input is a JSON list of hits or `{"hits": [...]}` from `--file` or stdin.
Human output has one line per hit plus one summary line. Exit code 0 means a
normal audit; exit code 2 means empty input, an unreadable file, invalid JSON,
or invalid thresholds/`now`.

## Scope and honest limits

Dates come only from the hit text, never from file modification time, file
system metadata, retrieval time or a model. This is deterministic evidence,
not a truth or freshness guarantee: an old date does not prove a claim false,
a recent date does not prove it true, and an `undated` source is not evidence
of recency. The signal is only as good as the dates authors wrote in the
source.

```powershell
$env:PYTHONPATH="$PWD\src"
python -m pytest tests/test_freshness.py -q
python -m aisc check docs/aisc/atomzero-freshness.toml --strict
```
