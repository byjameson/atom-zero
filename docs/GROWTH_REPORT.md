# Growth report

`atomzero-growth` is a single, content-free view of the growth ladder. It
combines local evidence that already exists; it does not call a model, teacher
or agent.

```powershell
atomzero-growth `
  --escalation-jsonl data/escalation_growth.jsonl `
  --scoreboard-jsonl data/scoreboard.jsonl `
  --self-code-diff-json data/self_code_diff.json `
  --json
```

## Inputs

- `--escalation-jsonl`: content-free events written by
  `atomzero-escalate` (`record_escalation`).
- `--scoreboard-jsonl`: measured runs from `atomzero-scoreboard`.
- `--self-code-diff-json`: a `atomzero-self-code-diff` report.

At least one source is required. Only one source may use `-` for stdin.

## Output

The report contains:

- route counts: local, teacher, agent, refuse;
- verified teacher answers and adapter candidates;
- latest test, benchmark and AISC counts plus the scoreboard delta;
- self-code resolved/introduced/persisting counts;
- `parameter_growth: 0` and `parameter_growth_state: unchanged`;
- `weights_changed: false` and `auto_apply: false`.

`ok` is true only when:

- a scoreboard run exists and has zero failed tests;
- the supplied self-code diff has `ok: true`;
- no other blocking reason is present.

## Honest boundary

The report measures capability and code health. It does not prove product
value, does not train an adapter and does not change a parameter count. A
verified teacher answer becomes an **adapter candidate**, not a trained
adapter. Parameter growth still requires the offline, human-approved training
and evaluation steps described in `docs/GROWTH_LADDER.md`.
