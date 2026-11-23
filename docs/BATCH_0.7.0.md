# Batch 0.7.0 - ten increments before the next push

The owner asked for a push at roughly ten development increments, not after
every version. `0.6.1` is the last published tag. The following work accumulated
on local `main` and is now at the `0.7.0` boundary: **all ten increments are
merged, 646 Atom Zero tests pass, and AISC 0.5.1 has 223 tests.**

## Increments

| # | Increment | Status |
|---|---|---|
| 1 | `POST /v1/claims/audit` | merged on local main |
| 2 | Cross-source consensus (`atomzero-consensus`, `POST /v1/consensus`) | merged on local main |
| 3 | Aggregate `atomzero-audit --summary` | merged on local main |
| 4 | Freshness algebra + `POST /v1/freshness` | merged on local main |
| 5 | Content-free source reliability ledger + HTTP summary/verify | merged on local main |
| 6 | AISC `batch --baseline` regression gate | implemented in AISC 0.5.1 |
| 7 | Document version diff (`atomzero-doc-diff`, `POST /v1/documents/diff`) | merged on local main |
| 8 | Temporal validity / as-of filtering (`POST /v1/temporal/filter`) | merged on local main |
| 9 | Claim-aware proactive conversation | merged on local main |
| 10 | AISC spec coverage report (`atomzero-coverage`) | merged on local main |

The research scan behind these choices is in `docs/BRANCH_RESEARCH.md`.

## Release boundary

At the boundary:

1. Reviewer runs full pytest, strict AISC batch, build/twine and the focused
   branch tests on the merged tree.
2. Bump the package version and update `CHANGELOG.md`, `README.md`, release
   notes, Hugging Face and ModelScope cards.
3. Push `main` and the tag once, create the GitHub release and upload both
   assets.
4. Upload to PyPI with the owner's project token.
5. Update dev.to, Medium and LinkedIn once.
6. Run `scripts/verify_release.py --version <version>` and have the read-only
   Release agent confirm every surface.

Do not push intermediate commits. A hotfix is the only exception and still
gets one push, not one per commit.
