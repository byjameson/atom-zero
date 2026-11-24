# Ubuntu VM test sandbox

Risky or long-running verification is executed inside the Ubuntu 26.04 VMware
VM, not on the Windows host.

## Current VM

```text
VM:            Ubuntu 64-bit (VMware Workstation)
IP:            192.168.121.128
CPU / RAM:     2 vCPU / 6 GB
Guest user:    byjameson
SSH:           key-based, port 22
Python:        3.12 in ~/atom-zero/.venv
torch:         2.15.0+cpu
```

The password is not stored in the repository. SSH uses the host key
`~/.ssh/id_ed25519`.

## Run the full suite in the VM

```powershell
powershell -File scripts/vm_test.ps1
```

Sync the local `src`, `tests`, `research` and `pyproject.toml` first:

```powershell
powershell -File scripts/vm_test.ps1 -Sync
```

The script runs:

```bash
cd ~/atom-zero
PYTHONPATH=src .venv/bin/python -m pytest tests -q
PYTHONPATH=src .venv/bin/python benchmarks/self_improvement_regression.py
```

## Current VM evidence

```text
pytest tests -q:  1130 passed
benchmark:        20/20
```

## Notes

- The VM has no GPU; LoRA training runs on CPU and is slower but safe.
- The host GPU is used only for teacher generation/data preparation.
- The VM is a disposable environment. Base checkpoints and teacher fixtures are
  copied locally; nothing is pushed from the VM.
- Do not commit the guest password or SSH private key.
