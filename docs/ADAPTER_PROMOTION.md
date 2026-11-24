# Gated adapter promotion

A candidate is promoted only when the full-loss gate, the content-loss gate
and the behavior gate (exact match and key copy) all pass. Promotion writes a
local active-adapter registry; it does not change base `0.8.0` weights.

A format-only adapter cannot be promoted: passing loss and content metrics
without `exact_match_ok` and `key_copy_ok` reasons is rejected.

```powershell
atomzero-adapter-candidate promote `
  --candidate-json data/diverse_autonomous_replay/candidate.json `
  --gate-json research/results/replay_cross_pass.json `
  --adapter-dir checkpoints/adapters_replay/v0001 `
  --active-file data/active_adapter.json --json
```

The command validates:

- the gate status is `pass`, `ok: true` and `promotable: true`;
- `content_loss_improved` and `content_ppl_not_worse` are present;
- `exact_match_ok` and `key_copy_ok` are present;
- `adapter_exact_match` is at least `min_exact_match` when the gate records
  that threshold;
- the gate candidate id matches the candidate metadata;
- `family_separation: true` is present;
- `adapter.pt` exists;
- the result is written atomically to the active file.

A legacy gate that only carries the content reasons fails with a `ValueError`
that names the missing behavior reasons. A gate without `exact_match_ok` or
`key_copy_ok` is never promoted, even if its loss numbers look good.

The registry is content-free apart from the local adapter path:

```json
{
  "path": "checkpoints/adapters_replay/v0001",
  "candidate_id": "ad-e396366ef725656c",
  "gate_status": "pass",
  "promotable": true,
  "family_separation": true,
  "adapter_bytes": 1482075,
  "promoted_at": "2026-11-24T01:08:22Z",
  "previous": "",
  "weights_changed": false,
  "auto_apply": false,
  "method": "gated_adapter_promotion_v1"
}
```

## Product activation

```powershell
atomzero --adapter auto --active-adapter-file data/active_adapter.json
```

`chat_server.resolve_adapter("auto", active_file)` reads the local registry and
applies the adapter with `atomzero.lora.apply_adapter`. The base checkpoint
remains `0.8.0`.

## Rollback

Delete `data/active_adapter.json`, or point `path` back to the `previous`
adapter. Restart the server. Rejected candidates never touch the registry.

## Removed promotion - content-only

The replay-weighted candidate was retrained with content-weighted loss
(`--content-weight 5`) and passed the loss and content gates only:

```text
candidate:          ad-e396366ef725656c
full loss:          5.2549 -> 2.8489   (pass)
full ppl ratio:     0.0902
content loss:       4.4327 -> 3.4892   (delta -0.9435, required -0.05; pass)
content ppl ratio:  0.3893
family separation:  true
result:             promotable
adapter bytes:      1,482,139
weights_changed:    false
```

The registry was removed after the behavior audit: the adapter had no
held-out exact matches. The base checkpoint remains `0.8.0`; no adapter is
active.

## Behavior gate - required for promotion

Content-token loss is still a token-average metric. The exact-match audit in
`docs/CONTENT_AUDIT.md` shows the removed registry entry at **0/48** on the
held-out city family and 5/28 on the in-distribution entities. It copied
values and formats, not unseen key names. The gate therefore fails when run as:

```powershell
atomzero-adapter-candidate evaluate `
  --candidate-json data/diverse_autonomous_content/candidate.json `
  --comparison-json research/results/content_adapter_cross_audit.json `
  --require-content --min-exact-match 0.25 --min-key-copy-rate 0.25 --json
```

The `evaluate` and `loop` subcommands both accept `--min-key-copy-rate`;
`atomzero-self-improve` forwards `--adapter-min-key-copy-rate`. The key-copy
metric is read from `adapter.key_copy.rate` or `adapter.key_copy_rate` in the
comparison report.

Promotion now requires the full loss gate (`loss_improved`,
`ppl_not_worse`), the content gate (`content_loss_improved`,
`content_ppl_not_worse`), exact match (`exact_match_ok`) and key copy
(`key_copy_ok`). The evaluation thresholds stay opt-in, but `promote` always
enforces the resulting reasons and re-checks `adapter_exact_match` against
`min_exact_match` when present.

No registry entry is active. The earlier content-only registry entry was
removed because it could not pass the behavior gate.

