# Releasing Atom Zero

Canonical surfaces: GitHub, Hugging Face, PyPI, dev.to, Medium, LinkedIn and
ModelScope.

Current batch: **0.30.0 published. Next boundary: 0.40.0.** It adds the bounded self-improve loop,
committed-change scope verification and the fifth measured cycle with an actual
nested `opencode run` process. Post-release work adds the self-code audit, its
planner/runner integration and the sixth measured cycle.
`1303` Atom Zero tests pass; AISC 0.5.1 has `223` tests and
`benchmarks/self_improvement_regression.py` passes 20/20.

## Release cadence

Releases happen **once every 10 minor versions**: `0.20.0 -> 0.30.0 ->
0.40.0` and so on. Do not publish `0.21.x`-`0.29.x` as release boundaries.
Intermediate patch uploads may exist only as hotfixes and are not the release
milestone.

`0.20.0` was the last 0.1 boundary; the next boundary is **`0.30.0`**. Every
0.1 boundary is published on all canonical surfaces.

Intermediate commits stay local. At every 0.1 boundary: bump the version,
update release notes/model cards/docs, run preflight and
`scripts/verify_release.py`, and publish every canonical surface once. Push the
release commit and tag, but do not push each increment.

## 1. Preflight

```powershell
python -m pytest tests -q
python -m build
python -m twine check dist/*
# Optional when AISC is available:
atomzero-audit --aisc-path "<aisc-checkout>" --strict --require-calibration
atomzero-doctor --aisc --aisc-path "<aisc-checkout>"
```

Expected for the next batch: `465 passed` or more.

## 2. PyPI

Use a fresh scoped API token supplied by the owner. Never reuse a token that
has appeared in chat or logs.

```powershell
python -m twine upload dist/atomzero-<version>-py3-none-any.whl dist/atomzero-<version>.tar.gz
```

Check that the JSON description is clean:

```text
https://pypi.org/pypi/atomzero/<version>/json
```

## 3. GitHub

- Commit and push `main`.
- Tag `v<version>`.
- Create a GitHub release.
- Attach the wheel and sdist.

## 4. Hugging Face

Update the model card on
<https://huggingface.co/byjameson/atom-zero-49m>. The card must contain the
current `atomzero` version and must not link to `stream-finetune` docs.

## 5. ModelScope

Update <https://modelscope.cn/models/byjameson/atom-zero-49m>.

- The browser session is logged in for UI checks.
- The README can also be uploaded through the ModelScope Hub API.
- The trainer mirror is <https://modelscope.cn/models/byjameson/streamft>.

## 6. Articles and posts

Publish a **new post** for every 0.1 release; do not silently edit the old one.

- dev.to: create a new post with the current install version and new commands.
- Medium: create a new post from the paste-ready source/HTML.
- LinkedIn: create a new post for every 0.1 release.
- X: post a short release note with the main links.
- Instagram: post a short release note or carousel with the main links.
- Include the author profiles in every post:
  - X: <https://x.com/aigokhan>
  - Instagram: <https://instagram.com/quantsinner>

## 7. Verify all surfaces

```powershell
python scripts/verify_release.py --version <version> --devto-url <url>
```

The script checks PyPI, Hugging Face, ModelScope, GitHub release assets and the
GitHub README, and optionally a dev.to article. It exits non-zero on a mismatch.

## 8. Bridge the trainer

`streamft` is a separate package. Do not add `atomzero` back into its wheel or
description. The trainer repo stays trainer-only.
