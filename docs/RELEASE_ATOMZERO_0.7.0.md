# atomzero 0.7.0 - evidence and temporal loop

Status: **published.** PyPI `0.7.0`, GitHub source/tag/release `v0.7.0`,
Hugging Face, ModelScope, dev.to, Medium and LinkedIn are updated.
`scripts/verify_release.py --version 0.7.0` reports **0 failures**. This is the
ten-increment batch accumulated after `0.6.1`. No model or runtime weights
changed; the weights remain `0.8.0`.

## What is in this batch

1. **Claim audit over HTTP.** `POST /v1/claims/audit` runs the deterministic
   claim/evidence audit without persisting the query or hits.
2. **Cross-source consensus.** `atomzero-consensus` and `POST /v1/consensus`
   cluster claims from multiple hits, label single-source clusters and keep
   numeric/date/identifier/negation conflicts separate.
3. **Aggregate AISC summary.** `atomzero-audit --summary` prints one
   pass/fail/error line across specs, with a JSON option.
4. **Freshness algebra.** `atomzero-freshness` and `POST /v1/freshness` read
   dates from hit text and classify fresh/aging/stale/future/undated.
5. **Content-free source reliability ledger.** `atomzero-reliability`,
   `POST /v1/reliability/summary` and `/verify` store hash-only counts and a
   support ratio; the writer verifies the ledger before an atomic rewrite.
6. **AISC batch baseline.** AISC 0.5.1 `aisc batch --baseline` classifies new
   failures, fixed specs, new/removed specs and errors.
7. **Document version diff.** `atomzero-doc-diff` and
   `POST /v1/documents/diff` report added/removed/changed blocks and a
   similarity ratio.
8. **Temporal validity.** `atomzero-temporal` and
   `POST /v1/temporal/filter` apply explicit `valid_from`/`valid_until`
   windows for an as-of date and report included/excluded evidence.
9. **Claim-aware proactive conversation.** Claim/consensus audits can record
   content-free counts on a local session; opt-in proactive check-ins can start
   from unsupported claims, conflicts or single-source clusters.
10. **AISC coverage report.** `atomzero-coverage` and
    `atomzero-audit --coverage` check component verification, evidence-file
    existence and calibration counts across the specs.

## Verified

- **646 Atom Zero tests passed** at the release commit.
- **AISC 0.5.1: 223 tests passed** (+6 subtests), wheel and sdist build and
  `twine check` pass.
- AISC strict: **14 specs, 12 PASS, 2 expected answerability failures**
  (`AtomZeroAnswerability` and its generated mirror). The failures are retained
  with Brier 0.244 / ECE 0.130 and refusal-policy misses.
- AISC coverage: **14 specs clean**, 0 missing evidence files, 0 unverified
  components; uncalibrated components remain visible.
- `scripts/verify_release.py --version 0.7.0`: **0 failures** (PyPI, HF,
  ModelScope, GitHub release/assets, GitHub README, dev.to).
- Release commit `72c9b5b`, annotated tag `v0.7.0`.

## Install

```powershell
pip install atomzero==0.7.0
atomzero-quickstart
```

## Honest limits

- Claim, consensus, reliability and freshness checks are lexical/deterministic
  audit signals, not truth or NLI verdicts.
- Consensus is lexical agreement, not independent verification.
- Reliability is a small-sample local ratio, not source credibility.
- Temporal filtering trusts `valid_from`/`valid_until`; it does not infer
  validity from prose.
- Proactive claim check-ins are opt-in, local and rate-limited; the claim
  signal stores counts only.
- The answerability gate remains off; CitationGate stays at `0.5`.
- AISC coverage is not assurance and does not lower the failing thresholds.

## Support

- **EVM-compatible networks:** `0xa4dEc84Fe6F13c8A3b8707446D6892a6E272285c`
- **Solana:** `EDqs8w4JkS5vAAfxmsB7fAQwoauEYufYXvMWFW9aYPdc`
- **Bitcoin:** `bc1qsq4mdtf0dlfvppllqwdaknxrq6xkjgsrx6ctgm`

Full details: <https://github.com/byjameson/atom-zero/blob/main/docs/DONATE.md>

## Links

- PyPI: <https://pypi.org/project/atomzero/0.7.0/>
- GitHub: <https://github.com/byjameson/atom-zero>
- GitHub release: <https://github.com/byjameson/atom-zero/releases/tag/v0.7.0>
- Branch research: <https://github.com/byjameson/atom-zero/blob/main/docs/BRANCH_RESEARCH.md>
- Ten-increment plan: <https://github.com/byjameson/atom-zero/blob/main/docs/BATCH_0.7.0.md>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- Contact: **gokhancava@gmail.com**
