# Releasing Atom Zero

Canonical surfaces: GitHub, Hugging Face, PyPI, dev.to, Medium, LinkedIn and
ModelScope.

Current batch: **0.19.0 published.** It adds the bounded self-improve loop,
committed-change scope verification and the fifth measured cycle with an actual
nested `opencode run` process. Post-release work adds the self-code audit, its
planner/runner integration and the sixth measured cycle.
`1032` Atom Zero tests pass; AISC 0.5.1 has `223` tests and
`benchmarks/self_improvement_regression.py` passes 19/19.

## Release cadence

Releases follow **one major version per release** now, not every 0.1 or 0.3.
`0.19.0` is the current published version; the next release boundary is
**`1.0.0`**, then `2.0.0`. Do not publish `0.20.x`–`0.99.x`.

Intermediate commits stay local. A release batch can contain many development
increments, but it is only published at a major boundary. A hotfix is the only
exception and still gets one push.

At the batch boundary: bump the version, update release notes/model cards/docs,
run preflight and `scripts/verify_release.py`, and publish every canonical
surface once. Do not push each increment.

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

- dev.to: update the install version and any new command.
- Medium: update the source and paste-ready HTML.
- LinkedIn: update the post only for meaningful releases.

## 7. Verify all surfaces

```powershell
python scripts/verify_release.py --version <version> --devto-url <url>
```

The script checks PyPI, Hugging Face, ModelScope, GitHub release assets and the
GitHub README, and optionally a dev.to article. It exits non-zero on a mismatch.

## 8. Bridge the trainer

`streamft` is a separate package. Do not add `atomzero` back into its wheel or
description. The trainer repo stays trainer-only.
