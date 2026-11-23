# Document version diff

Local RAG documents change. `atomzero.version_diff` compares two text versions
at paragraph-block level and reports added, removed and changed blocks plus a
similarity ratio. It is stdlib-only, deterministic and does not modify the
document memory.

```powershell
$env:PYTHONPATH="$PWD\src"
python -m atomzero.version_diff --old old.txt --new new.txt --json
```

```python
from atomzero.version_diff import audit_document_diff

report = audit_document_diff(old_text, new_text)
# {"changes": {"added": [...], "removed": [...], "changed": [...]},
#  "summary": {...}, "weights_changed": False,
#  "method": "deterministic_block_diff_v1"}
```

## Blocks

- Text is normalized to LF and split on blank lines.
- Horizontal whitespace is collapsed; leading/trailing whitespace is removed.
- Empty blocks are dropped.
- The diff uses `difflib.SequenceMatcher`; `similarity` is its ratio rounded
  to four decimals.
- A `changed` pair is a replaced block at the same position; extra blocks on
  either side become added/removed.

## HTTP

`POST /v1/documents/diff` accepts `old_text` and `new_text` (each limited to
200,000 characters), plus optional `old_label` and `new_label`. The route is
stateless and does not store either text.

```powershell
$body = @{
  old_text = "The price is 10."
  new_text = "The price is 12."
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/documents/diff `
  -ContentType "application/json" -Body $body
```

## Honest limits

- Paragraph comparison is not semantic versioning. A reordered document can
  look mostly changed; a one-word edit can look like a whole changed block.
- The report does not decide which version is newer or more correct. Labels are
  caller-provided.
- It does not update the index, re-embed chunks or supersede old sources.
- Empty comparisons are defined: two empty texts are unchanged with similarity
  `1.0`; one empty side gives `0.0`.
- `weights_changed` is always false; no model weights are touched.

## Verification

```powershell
$env:PYTHONPATH="$PWD\src"
python -m pytest tests/test_version_diff.py tests/test_chat_server_docs.py::test_http_document_diff -q
python -m aisc check docs/aisc/atomzero-document-diff.toml --strict
```

The AISC spec declares three verified components, no irreversible steps and
`min_assurance = 0.90`.
