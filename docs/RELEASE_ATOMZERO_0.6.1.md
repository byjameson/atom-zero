# atomzero 0.6.1 - claim evidence and generated AISC specs

Status: **published.** PyPI `0.6.1`, GitHub source/tag/release `v0.6.1`,
Hugging Face, ModelScope, dev.to, Medium and LinkedIn are updated.
`scripts/verify_release.py --version 0.6.1` reports **0 failures**. No model or
runtime weights changed; the weights remain `0.8.0`.

This is a post-`0.6.0` increment batch. It brings the memory lifecycle ledger
and two new deterministic branches into the published package.

## Claim / Evidence Algebra

- `atomzero-claims`, `from atomzero.claims import audit_answer` and
  `python -m atomzero.claims` split text into sentence-level claims with stable
  IDs and offsets.
- Each claim is matched to the best retrieved hit with a transparent lexical
  term-coverage/overlap score (`supported`, `weak`, `unsupported`). The
  attached evidence keeps source, title, section, page and snippet.
- `compare_claims` flags number, unit, date, identifier and explicit negation
  conflicts between claims using the existing transparent normalisation.
- This is an audit signal, not a natural-language-inference or truth model. It
  makes no model calls and does not change retrieval or weights.
- See `docs/CLAIMS.md` and `docs/aisc/atomzero-claims.toml`.

## Generated AISC specs

- `atomzero-aisc-spec` reads
  `benchmarks/results/answerability_hybrid/answerability_profile.json`,
  validates the required evidence keys and emits a canonical TOML spec.
- `--check` performs byte-level drift detection without writing; writes are
  atomic.
- The generator never imports AISC and never lowers target thresholds. The
  generated spec keeps the measured answerability failures visible, so strict
  AISC still reports `ANSWERABILITY_POOR` and `REFUSAL_POLICY_MISMATCH`.
- Generated artifact: `docs/aisc/atomzero-answerability-generated.toml`.
- See `docs/AISC_SPEC_GENERATOR.md`.

## Memory lifecycle ledger

- `atomzero-memory-ledger`, `GET /v1/memory/ledger` and
  `POST /v1/memory/ledger/verify`.
- Successful document/session deletes append counts and scope only; the
  validator rejects text, query, answer, chunk and session content fields.
- Direct HTTP deletion requires an explicit server-side `confirm: true`.

## AISC strict status

- `AtomZeroClaims`: **PASS** (4 components, assurance bound 0.93, 0
  irreversible steps).
- Read, task, tool-plan, context-budget, proactive and memory-lifecycle specs:
  **PASS**.
- `AtomZeroAnswerability` and its generated mirror
  `AtomZeroAnswerabilityGenerated`: **FAIL** by design of the measurement, not
  hidden. Brier 0.244 and ECE 0.130 miss the 0.15/0.10 bounds; refusal recall
  0.546, precision 0.631 and answerable coverage 0.699 miss the
  0.90/0.90/0.80 targets. The gate remains off and CitationGate stays at 0.5.
- Total at the `0.6.1` tag: 9 specs, **7 PASS**, 2 expected answerability
  failures.
- `--require-calibration` still reports `CALIBRATION_MISSING` for specs without
  target-distribution calibration data. This is not claimed as green.

## Verified regressions

- **512 Atom Zero tests passed** at the release commit.
- `python -m build` and `twine check` passed for the wheel and sdist; both
  `0.6.1` artifacts are on PyPI and the GitHub release.
- AISC strict: 7 specs PASS; the answerability spec and its generated mirror
  remain FAIL with 0 critical and 2 expected warnings.
- `scripts/verify_release.py --version 0.6.1`: **0 failures**.
- Full details: `TEST_RESULTS.md`.

## Install

```powershell
pip install atomzero==0.6.1
atomzero-quickstart
```

## Honest limits

- Claim linking is lexical, not entailment. Paraphrases can look unsupported;
  copied phrases can look supported even with wrong numbers.
- `compare_claims` is a phrase-level audit, not a truth verdict.
- The AISC spec generator trusts the stored profile; it does not rerun the
  answerability benchmark.
- AISC calibration remains unproven for the configured targets.

## Support

- **EVM-compatible networks:** `0xa4dEc84Fe6F13c8A3b8707446D6892a6E272285c`
- **Solana:** `EDqs8w4JkS5vAAfxmsB7fAQwoauEYufYXvMWFW9aYPdc`
- **Bitcoin:** `bc1qsq4mdtf0dlfvppllqwdaknxrq6xkjgsrx6ctgm`

Full details: <https://github.com/byjameson/atom-zero/blob/main/docs/DONATE.md>

## Links

- PyPI: <https://pypi.org/project/atomzero/0.6.1/>
- GitHub: <https://github.com/byjameson/atom-zero>
- GitHub release: <https://github.com/byjameson/atom-zero/releases/tag/v0.6.1>
- Claims: <https://github.com/byjameson/atom-zero/blob/main/docs/CLAIMS.md>
- AISC spec generator: <https://github.com/byjameson/atom-zero/blob/main/docs/AISC_SPEC_GENERATOR.md>
- Memory lifecycle: <https://github.com/byjameson/atom-zero/blob/main/docs/MEMORY_LIFECYCLE.md>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- Contact: **gokhancava@gmail.com**
