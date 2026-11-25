# Vast.ai experiment infrastructure

Status: infrastructure ready. **No instance is rented by default.**
Training and large-model work belong on a rented GPU; tests and gates stay on
the VMware Ubuntu CPU box.

## When to rent

Rent only when all three are true:

1. the experiment does not fit the local RTX 3050 6 GB, or would take many
   hours locally;
2. there is at least **30–60 minutes** of useful GPU work (the launcher refuses
   less than 30 planned minutes);
3. the run has a written hypothesis and a content-free expected output.

Do not rent for a 5-minute check, a test run, or a job that is really CPU-bound
tokenization. Those stay local.

## Budget and cost guards

The account currently shows **$10.00 credit** (`balance 0` + `credit 10`).
Live offers at setup time:

```text
RTX 3090 24 GB   $0.11–0.25/hr
RTX 4090 24 GB   $0.37–0.47/hr
```

A 3-hour 3090 run is roughly **$0.33–0.75** of compute. The launcher defaults
to a 3-hour hard cap, refuses a worst-case estimate above $9, and destroys the
instance in `finally` unless `--keep` is passed explicitly. A kept instance
prints a warning and must be destroyed manually.

## Files

| file | purpose |
|---|---|
| `scripts/vastai/preflight.py` | read-only: balance, credit, suitable offers, cost estimate |
| `scripts/vastai/bundle.py` | reproducible `.tar.gz` of code + exact data files + SHA-256 manifest |
| `scripts/vastai/job.sh` | on-instance: download base, upcycle, full fine-tune, copy + retention evals, summary |
| `scripts/vastai/launch.py` | dry-run by default; `--execute` rents, uploads, polls, downloads results, destroys |
| `scripts/vastai/common.py` | pure offer/SSH/cost helpers, unit-tested |

The API key is read at runtime from `Desktop/vastai.txt`. It is never printed
or written into the repository. The local SSH public key is registered with
Vast.ai so `scp`/`ssh` work; `vastai show ssh-keys` lists it.

## Standard workflow

### 1. Preflight

```powershell
python scripts/vastai/preflight.py --max-price 0.25 --hours 3
```

Pick a **24 GB** offer with reliability above 0.95 and at least 100 Mbps
download. Prefer the cheapest reliable 3090 for this workload; a 4090 is only
worth it when a run is latency-bound.

### 2. Build the bundle

```powershell
python scripts/vastai/bundle.py
```

Outputs `dist/vastai/atomzero-bundle.tar.gz` plus a sidecar with SHA-256, git
HEAD and a per-file manifest. Data included: the combined copy curriculum,
2/3/4-key held-out sets, mixed eval, rehearsal rows, city retention sets and
the out-of-distribution 4-key set. Checkpoints are never bundled; the instance
downloads the public 49M base from Hugging Face.

### 3. Dry run

```powershell
python scripts/vastai/launch.py --offer-id <OFFER_ID> --dph <PRICE> --layers 24
```

This prints the exact plan and the worst-case cost. Nothing is rented.

### 4. Execute

```powershell
python scripts/vastai/launch.py --offer-id <OFFER_ID> --dph <PRICE> `
  --layers 24 --seeds "101 102" --steps 4000 --batch 16 --max-hours 3 --execute
```

The launcher:

1. refuses if another `atomzero-` instance is running;
2. creates an SSH instance with the `pytorch/pytorch:2.5.1-cuda12.4` image and
   50 GB disk;
3. uploads the bundle, starts `job.sh` under `nohup`;
4. polls `out/DONE` / `out/FAILED` every minute;
5. downloads `out/` to `dist/vastai/runs/<UTC timestamp>/`;
6. **destroys the instance** and prints elapsed time plus estimated compute.

Expected outputs: `summary.json`, `upcycle_24l.json`, full fine-tune reports,
`*_2short|3short|4short.json`, `*_retention.json` and `job.log`.

### 5. After the run

- Copy the content-free summary into `docs/` or a committed result file.
- Run the VM suite and the checkpoint gate before any claim.
- If the run failed: read `job.log`; do not silently retry a paid run.

## Manual fallback

If the launcher fails after the instance exists, use the CLI directly:

```powershell
$env:VAST_API_KEY = (Get-Content "$env:USERPROFILE\Desktop\vastai.txt" -Raw).Trim()
vastai show instances --raw
vastai ssh-url <ID>
scp -P <PORT> dist/vastai/atomzero-bundle.tar.gz root@<HOST>:/workspace/
# then start job.sh over ssh, or use: vastai execute <ID> 'bash ...'
vastai destroy instance <ID> -y
```

Never leave an instance running while debugging locally.

## Experiment definition (first paid run)

Question: does the 4-key copy gain scale with depth, and what does it cost in
content retention?

```text
base:            49M / 0.8.0
upcycle:         24 layers (~177M), interpolate, seed 23
training:        copy curriculum + 153 verified teacher rows, 4000 steps,
                 batch 16, lr 5e-5, content weight 2
seeds:           101, 102
evaluation:      200 held-out rows per 2/3/4-key stage
retention:       49M baseline vs 24L candidate on city YAML/river, max 200 rows
out-of-dist:     30 fresh 4-key rows with field_/k_ styles
```

Success is not just a higher copy number: the run must produce a retention
ratio and the checkpoint gate must be reviewable before any product claim.

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `no matching offers` | `gpu_ram` query is in **GB**, not MB | keep `--min-vram 24`; raise `--max-price` |
| `Permission denied (publickey)` | no SSH key on the account | `vastai create ssh-key (Get-Content ~/.ssh/id_ed25519.pub -Raw) -y` |
| instance stays `loading` | host scheduling | wait up to 15 min; the launcher aborts after that |
| CUDA OOM | batch too high for the chosen card | rerun with `--batch 8`; 24 GB should handle 16 at 177M |
| job dies after training | evaluation OOM or missing data | inspect `out/job.log`; bundle manifest lists every file |
| results missing | `scp` failed | instance is destroyed anyway; rerun with `--keep` only for debugging |

## Rules

- The product base remains `0.8.0` until a candidate passes the checkpoint
  gate, retention and the full VM suite.
- No secrets on the rented box. `HF_TOKEN`/`HF_REPO` are opt-in and off by
  default.
- Destroy the instance even when the job fails.
- Release only at the 10-minor boundary, with fresh posts on every surface.
