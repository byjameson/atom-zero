# Source Reliability Ledger

`atomzero.reliability` is a small, standalone ledger for recording how an
upstream claim audit scored observations of a source. It is deterministic and
local: it makes no model calls, opens no network connections, and only touches
the file it is explicitly given.

The ledger answers a narrow question: *for this source label, how often did the
local audit find claims supported, weak, or unsupported?* It does not decide
whether a source is true.

## Content-free boundary

The source label is hashed and immediately discarded:

```python
source_id("https://example.org/article")[:16]
# stable 16-character SHA-256 prefix
```

Entries contain only:

```json
{
  "version": 1,
  "at": "2026-11-23T12:00:00Z",
  "source_id": "5d41402abc4b2a76",
  "supported": 3,
  "weak": 1,
  "unsupported": 1,
  "claims": 5,
  "conflicts": 0
}
```

They never contain the source label, source text, query, answer, chunk,
session ID, prompt, or any other content. `verify()` recursively rejects the
keys `text`, `query`, `answer`, `chunk`, `content`, `session`, `session_id`,
`source`, and `source_text`.

The hash is a stable pseudonym, not anonymization. A short label such as a
common domain can be guessed by hashing candidate labels, and a truncated hash
can collide in principle. The ledger deliberately stores no salt, so equal
trimmed labels always map to the same `source_id`.

## Observation validation

`add_observation()` requires:

- `supported`, `weak`, `unsupported`, and `conflicts` are non-negative
  integers (booleans do not count);
- at least one claim in total;
- `conflicts <= supported + weak + unsupported`.

`claims` is computed as `supported + weak + unsupported`. The timestamp is
normalized to ISO-8601 UTC with a trailing `Z`. Invalid input raises
`ValueError` before anything is appended.

## Score

For each source, the summary uses:

```
support_ratio = (supported + 0.5 * weak) / claims
```

Weak claims count half. Unsupported claims count zero. The stored value is
rounded to four decimal places, while the status threshold is applied to the
unrounded ratio:

| Status | Unrounded ratio |
| --- | --- |
| `reliable` | `>= 0.75` |
| `mixed` | `>= 0.40` |
| `weak` | `< 0.40` |

Only sources with at least `min_observations` observations are included in the
summary. Totals are computed over the included sources. Sources are sorted by
`source_id`, so input order never changes the output.

`conflicts` are reported but are not subtracted from the ratio. They are an
audit hint, not a truth penalty.

## Python API

```python
from atomzero.reliability import (
    add_observation, summarize, verify, load_jsonl, write_jsonl)

ledger = load_jsonl("data/source_reliability.jsonl")
entry = add_observation(
    ledger, "example.org",
    supported=3, weak=1, unsupported=1, conflicts=0)
write_jsonl("data/source_reliability.jsonl", ledger)

print(summarize(ledger)["sources"])
print(verify(ledger))
```

`load_jsonl()` treats a missing file as an empty ledger and rejects malformed
JSON, non-object lines, and duplicate JSON keys. `write_jsonl()` writes UTF-8
JSONL to a temporary file in the same directory and atomically replaces the
target with `os.replace()`, so a failed write does not leave a partial ledger.
A timezone-aware `now` value is converted to UTC; a naive `now` value is
treated as UTC. Verification itself requires an explicit zero UTC offset.

`verify()` returns:

```json
{"ok": true, "entries": 2, "errors": []}
```

It checks required and unexpected keys, the version, the timestamp, the source
ID shape, count types and ranges, the claim sum, and the conflict bound. It
never raises for malformed entries; inspect `ok` and `errors`.

## Command line

```console
python -m atomzero.reliability --file ledger.jsonl \
    --add-source example.org --supported 3 --weak 1 --unsupported 1

python -m atomzero.reliability --file ledger.jsonl --summary
python -m atomzero.reliability --file ledger.jsonl --verify
python -m atomzero.reliability --file ledger.jsonl --summary --json
```

- `--add-source` appends exactly one observation using the supplied counts.
- `--summary` prints grouped totals; with no action flag it is the default.
- `--verify` prints `{ok, entries, errors}` and exits 2 when invalid.
- `--json` prints machine-readable JSON.
- Invalid arguments, invalid counts, unreadable JSONL, or a failed
  verification exit with status 2. A normal add or summary exits with 0.

The default file is `data/source_reliability.jsonl`; nothing is written unless
`--add-source` is used.

## HTTP

`POST /v1/reliability/summary` accepts a content-free `entries` list (maximum
1000) and optional `min_observations`; `POST /v1/reliability/verify` returns
`{ok, entries, errors}` for the same input. Both routes are read-only and do
not persist or rewrite the ledger.

```powershell
$body = @{ entries = @(
  @{ version = 1; at = "2026-11-23T00:00:00+00:00";
     source_id = "e39538e7f27a7bf5"; supported = 4; weak = 1;
     unsupported = 1; claims = 6; conflicts = 0 }
) } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/v1/reliability/summary `
  -ContentType "application/json" -Body $body
```

## Honest limits

- **Small samples.** A source with one or two observations can jump between
  `reliable` and `weak`. Use `min_observations` and treat the ratio as a
  descriptive statistic, not a confidence estimate.
- **Lexical audit bias.** The counts come from the upstream claim audit, which
  may reward wording overlap rather than factual support. A source that copies
  the answer text, or a claim phrased unusually, can be scored incorrectly.
  The ledger aggregates those judgments; it cannot correct them.
- **Not source truth.** `reliable` means the local audit frequently found
  support, not that the source is correct, current, unbiased, or safe. A
  consistently wrong source can look reliable.
- **No conflict penalty in the score.** Conflicts remain visible in each row
  and in totals, but the ratio deliberately stays simple.
- **Pseudonymous labels only.** There is no reversal map and no signature.
  Anyone who can write the file can add observations; `verify()` checks shape
  and content-free rules, not authorship.
