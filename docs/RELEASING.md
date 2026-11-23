# Releasing Atom Zero

Canonical surfaces: GitHub, Hugging Face, PyPI, dev.to, Medium, LinkedIn and
ModelScope.

Current batch: **0.10.0 published.** It adds calibration-drift monitoring and
two measured closed cycles (HTTP flake and structured-token conflict noise).
`794` Atom Zero tests pass; AISC 0.5.1 has `223` tests and
`benchmarks/self_improvement_regression.py` passes 12/12.
`scripts/verify_release.py --version 0.10.0` reports **0 failures**. LinkedIn
remains blocked/unverified.

Post-release work starts the next batch. Do not push intermediate commits; the
next boundary is the next ten-increment batch.

## Release cadence

Atom Zero uses a batched release train: accumulate about **10 development
increments** on `main` before publishing. Do not upload every patch.

Between batch boundaries:

- keep changes under `CHANGELOG.md` → `Unreleased`;
- run tests and builds locally;
- do not update PyPI, GitHub releases, Hugging Face, ModelScope or dev.to;
- no version bump is required for every commit.

At the batch boundary:

- bump the version;
- update release notes, model card and current docs;
- run preflight and `scripts/verify_release.py`;
- publish every canonical surface once.

**Push policy:** do not `git push` each increment. Accumulate local commits and
push `main` once at the batch boundary, together with the version tag and
release. A hotfix is the only exception, and it still gets one push, not one per
commit.

`0.5.0` was the first batched release under this policy. It included the graph,
skills, chains, recipes, daily brief and goal planner accumulated after 0.4.2.

`0.6.0` is the second batch. It includes the weekly review/report archive,
task due dates/priorities/search/filters/exports, priority-first sorting,
source comparison, graph paths and reliability fixes accumulated after
`0.5.0`.

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
