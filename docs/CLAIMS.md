# Claim / Evidence Algebra

A small, standalone branch that audits an answer at the sentence level:

```
ClaimExtractor -> EvidenceLinker -> ContradictionEvaluator -> AnswerBuilder
```

Everything is deterministic and local. `atomzero/claims.py` uses only the
Python standard library and existing `atomzero` helpers
(`doc_memory.best_snippet`, `contradictions` normalisation). It never calls a
model, never changes retrieval weights and does not modify any product module.

## What is deterministic

- **Sentence extraction.** Claims are split on `.`, `!`, `?` followed by
  whitespace, or on newlines. Each claim keeps exact `start`/`end` offsets
  into the original text and a stable ID derived from its filtered position
  and a short hash of the normalised text. Empty fragments and fragments
  shorter than 12 characters or 2 words are skipped.
- **Evidence linking.** Every claim is compared with every hit using claim
  term coverage, exact token overlap and (when present) identifier coverage:

  ```
  support = 0.75 * coverage + 0.25 * overlap
  support = 0.55 * coverage + 0.25 * overlap + 0.20 * identifier_coverage
  ```

  The best hit wins; ties keep the earlier hit. `status` is `supported` when
  `support >= min_coverage` (default `0.45`), `weak` for a non-zero score
  below the threshold, and `unsupported` when no hit shares a claim term. The
  evidence item carries `source`, `title`, `section`, `page` and a
  `best_snippet` extract; it is `null` when unsupported.
- **Contradiction hints.** `compare_claims` reuses the number/unit/date/code
  and negation normalisation from `atomzero.contradictions`. A pair is only
  compared when it shares at least two terms, or an entity plus a term. It
  reports `numeric_mismatch`, `date_mismatch`, `identifier_mismatch` and
  `negation_mismatch` with both claim texts and IDs. At most 100 pairs are
  returned.

## What this is not

This is **not** a full natural-language-inference model. It has no semantic
entailment, no paraphrase model, no world knowledge and no truth verdict.
Overlap is lexical (with the small morphology allowance already used by
`doc_memory`), so a correct claim phrased with different vocabulary can look
unsupported, and a copied phrase can look supported even when the number is
wrong. Negation is detected only through explicit marker words. Sentence
splitting is regex-based and can mis-handle abbreviations, decimals and
punctuation without following whitespace. Use the output as an audit signal
for a human, not as an automatic fact-check.

## API

```python
from atomzero.claims import (
    extract_claims, link_evidence, compare_claims, audit_answer)

claims = extract_claims(text)
# [{"claim_id": "c001-1a2b3c4d", "text": "...", "start": 0, "end": 31}]

linked = link_evidence(claims, hits, min_coverage=0.45)
# claim dict + {"support": 0..1, "status": "...", "evidence": {...}|None}

comparison = compare_claims(linked)
# {"claim_count", "conflict_count", "conflicts", "counts",
#  "method", "weights_changed": False}

report = audit_answer(query, hits)
# {"query", "claims", "summary", "contradictions",
#  "weights_changed": False, "method": "deterministic_claim_evidence_v1"}
```

`hits` are normal retrieval hits with `chunk` plus optional `source`,
`title`, `section` and `page`. The functions do not mutate the inputs.

## CLI

```powershell
$env:PYTHONPATH="$PWD\src"
python -m atomzero.claims --text "The battery lasts 8 hours." --json
python -m atomzero.claims --file notes.txt
```

`--text` and `--file` are mutually exclusive; stdin is read when neither is
given. Files are read as UTF-8 (a BOM is tolerated). The CLI audits the text
without retrieval hits. Exit code is `0` for a normal audit, `2` for empty
input or an unreadable file. `--json` prints the full `audit_answer`
structure; the default output is a one-line-per-claim summary.

## Verification

```powershell
$env:PYTHONPATH="$PWD\src"
python -m pytest tests/test_claims.py -q
python -m aisc check docs/aisc/atomzero-claims.toml --strict
```

The AISC spec declares all four components `verified = true`, no irreversible
components, `min_assurance = 0.90`, and cites `tests/test_claims.py` as
evidence.
