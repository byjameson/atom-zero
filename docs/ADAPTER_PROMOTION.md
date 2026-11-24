# Gated adapter promotion

A candidate is promoted only when **both** the full-loss gate and the
content-loss gate pass. Promotion writes a local active-adapter registry; it
does not change base `0.8.0` weights.

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
- the gate candidate id matches the candidate metadata;
- `family_separation: true` is present;
- `adapter.pt` exists;
- the result is written atomically to the active file.

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

## Promotion audit - not promoted

The replay-weighted candidate passed the **full-loss** gate but failed the
content audit:

```text
candidate:          ad-8cf19d2b5840de59
full loss:          5.2549 -> 2.6002   (pass)
content loss:       4.4327 -> 4.4123   (delta -0.0205, required -0.05; fail)
content ppl ratio:  0.9797
result:             not promotable
```

The registry was removed after the audit. The adapter is a synthetic teacher
format-transfer candidate; the full-loss gain is largely syntax/format
learning. Do not promote it until a content-passing candidate exists.
