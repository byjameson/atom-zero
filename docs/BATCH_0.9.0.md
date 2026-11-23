# Batch 0.9.0 - measured self-improvement

`0.8.0` shipped the human-gated loop. `0.9.0` makes the loop **measured**:
approval requires a passing gate, actions carry suggested checks, and a
content-free scoreboard tracks whether the numbers actually improve.

## Increments

| # | Increment | Status |
|---|---|---|
| 1 | Gated approval (`record_gated_decision`) | merged on local main |
| 2 | Suggested `pytest`/AISC checks on every action | merged on local main |
| 3 | Content-free scoreboard (`atomzero-scoreboard`) | merged on local main |
| 4 | Measured regression signal in the improvement plan | merged on local main |
| 5 | Scoreboard HTTP summary/record routes | merged on local main |
| 6 | Scoreboard in the weekly review | merged on local main |
| 7 | Gate report archive linked to decisions | merged on local main |
| 8 | Improvement trend report (plan + decisions + scoreboard) | merged on local main |
| 9 | Source-review workflow link from weak-source actions | merged on local main |
| 10 | End-to-end self-improvement benchmark (signal -> gate -> decision -> score) | merged; 12/12 checks pass |

## Rules

- An approve decision without a passing gate is rejected.
- A scoreboard record is counts and a note hash only; no test output or text.
- A regression signal is higher priority than a new feature signal.
- A failed metric stays visible; the plan does not average it away.
- Nothing is auto-applied or trained.

## Boundary

The next push is the `0.9.0` release boundary after all ten increments are
reviewed. Intermediate commits stay local.
