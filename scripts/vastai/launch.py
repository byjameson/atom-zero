"""Launch, monitor and destroy a Vast.ai GPU run for Atom Zero.

Safety properties:
- dry-run by default; ``--execute`` is required to spend money
- refuses jobs shorter than ``--planned-minutes`` (default 30)
- refuses to start when another ``atomzero-`` instance is running
- always destroys the instance in ``finally`` unless ``--keep`` is explicit
- never prints the API key

Typical use
-----------
    python scripts/vastai/launch.py --offer-id 62329792 --dph 0.11
    python scripts/vastai/launch.py --offer-id 62329792 --dph 0.11 --execute

The run downloads the public 49M base on the instance, upcycles it to
``--layers``, fully fine-tunes on copy data, evaluates 2/3/4-key copy and
retention, downloads ``out/`` to ``dist/vastai/runs/<timestamp>/`` and destroys
the machine.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import estimate_cost, parse_ssh_url, planned_minutes_ok  # noqa: E402

DEFAULT_KEY_FILE = Path.home() / "Desktop" / "vastai.txt"
DEFAULT_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
DEFAULT_BUNDLE = Path("dist/vastai/atomzero-bundle.tar.gz")
LABEL = "atomzero-090"


def read_key(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"key file not found: {path}")
    key = path.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    if len(key) < 20:
        raise SystemExit("key file does not look like a single API key")
    return key


def run(args: list[str], *, key: str, check: bool = True, timeout: int = 120) -> str:
    exe = shutil.which("vastai")
    if not exe:
        raise SystemExit("vastai CLI not found. Install: python -m pip install vastai")
    proc = subprocess.run(
        [exe, "--api-key", key, *args],
        capture_output=True, text=True, timeout=timeout,
    )
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[:400]
        detail = detail.replace(key, "***")
        raise SystemExit(f"vastai {' '.join(args)} failed: {detail}")
    return proc.stdout


def instances(key: str) -> list[dict]:
    raw = run(["show", "instances", "--raw"], key=key)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else data.get("instances", [])


def create_instance(args, offer_id: int, key: str) -> int:
    cmd = [
        "create", "instance", str(offer_id),
        "--image", args.image,
        "--disk", str(args.disk),
        "--ssh", "--direct",
        "--label", LABEL,
        "--onstart-cmd", "sleep infinity",
        "--raw",
    ]
    raw = run(cmd, key=key)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {}
    new_id = payload.get("new_contract") or payload.get("id")
    if new_id:
        return int(new_id)
    # Fall back to the newest instance carrying our label.
    mine = [i for i in instances(key) if str(i.get("label", "")).startswith(LABEL)]
    if not mine:
        raise SystemExit(f"instance was created but its id could not be found: {raw[:300]}")
    return int(max(mine, key=lambda i: int(i.get("id", 0)))["id"])


def wait_running(instance_id: int, key: str, *, timeout_s: int = 900) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for inst in instances(key):
            if int(inst.get("id", -1)) != instance_id:
                continue
            status = str(inst.get("actual_status") or inst.get("status") or "")
            if status == "running" and inst.get("ssh_host"):
                return inst
            print(f"  status: {status or 'unknown'}", flush=True)
        time.sleep(15)
    raise SystemExit("instance did not become reachable within 15 minutes")


def ssh_target(instance_id: int, key: str) -> tuple[str, str, int]:
    raw = run(["ssh-url", str(instance_id)], key=key).strip().strip('"')
    return parse_ssh_url(raw)


def ssh_base(user: str, host: str, port: int) -> list[str]:
    return [
        # -T/-n: never allocate a tty and never hold the channel open on stdin.
        "ssh", "-T", "-n", "-p", str(port),
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=20",
        "-o", "ServerAliveInterval=30",
        f"{user}@{host}",
    ]


def scp_base(port: int) -> list[str]:
    return [
        "scp", "-P", str(port),
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=20",
    ]


def wait_ssh_ready(ssh: list[str], *, timeout_s: int = 600) -> None:
    """Wait until the SSH proxy accepts a command, not just until a host is assigned."""
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        proc = subprocess.run(
            [*ssh, "echo READY"], capture_output=True, text=True, timeout=40
        )
        if proc.returncode == 0 and "READY" in proc.stdout:
            return
        last = (proc.stderr or proc.stdout or "").strip()[-200:]
        print(f"  ssh not ready: {last}", flush=True)
        time.sleep(15)
    raise SystemExit(f"SSH did not become ready: {last}")


def remote(ssh: list[str], command: str, *, timeout: int = 120, check: bool = True) -> str:
    proc = subprocess.run([*ssh, command], capture_output=True, text=True, timeout=timeout)
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[:400]
        raise SystemExit(f"remote command failed: {command}\n{detail}")
    return proc.stdout


def destroy(instance_id: int, key: str) -> None:
    print(f"destroying instance {instance_id}...", flush=True)
    run(["destroy", "instance", str(instance_id), "-y"], key=key, check=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Vast.ai Atom Zero runner")
    parser.add_argument("--key-file", default=str(DEFAULT_KEY_FILE))
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--offer-id", type=int, default=0)
    parser.add_argument("--dph", type=float, default=0.2,
                        help="offer price per hour, used for the estimate")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--disk", type=int, default=50)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--seeds", default="101 102")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--train-data",
                        default="data/multi_short_v1/train_all_plus_distill.jsonl")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--copy-checkpoint", action="store_true")
    parser.add_argument("--max-hours", type=float, default=3.0)
    parser.add_argument("--planned-minutes", type=float, default=60.0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--execute", action="store_true",
                        help="actually rent. Without it this is a dry run.")
    parser.add_argument("--keep", action="store_true",
                        help="do not destroy (debug only; costs money)")
    parser.add_argument("--out-root", default="dist/vastai/runs")
    args = parser.parse_args(argv)

    key = read_key(Path(args.key_file))
    bundle = Path(args.bundle)
    if not args.execute:
        if not bundle.exists():
            print("WARNING: bundle missing; run python scripts/vastai/bundle.py first")
        est = estimate_cost(args.dph, args.max_hours)
        print("DRY RUN - nothing will be rented")
        print(f"  bundle:          {bundle}")
        print(f"  offer:           {args.offer_id or '<required with --execute>'}")
        print(f"  image:           {args.image}")
        print(f"  disk:            {args.disk} GB")
        print(f"  layers/seeds:    {args.layers} / {args.seeds}")
        print(f"  steps/batch:     {args.steps} / {args.batch}")
        print(f"  data:            {args.train_data}")
        print(f"  cache:           {args.cache_dir or 'none'}")
        print(f"  max hours:       {args.max_hours:.1f} at ${args.dph:.2f}/hr")
        print(f"  worst-case cost: ${est:.2f} (compute only)")
        print(f"  planned minutes: {args.planned_minutes:.0f}")
        print()
        print("add --execute to rent and run; the instance is destroyed automatically")
        return 0

    if not args.offer_id:
        raise SystemExit("--offer-id is required with --execute")
    if not bundle.exists():
        raise SystemExit(f"bundle not found: {bundle} (run scripts/vastai/bundle.py)")
    if not planned_minutes_ok(args.planned_minutes, minimum=30.0):
        raise SystemExit("refusing to rent for less than 30 planned minutes")
    est = estimate_cost(args.dph, args.max_hours)
    if est > 9.0:
        raise SystemExit(f"worst-case estimate ${est:.2f} is too close to the budget")

    running = [i for i in instances(key) if str(i.get("label", "")).startswith("atomzero-")]
    if running:
        ids = ", ".join(str(i.get("id")) for i in running)
        raise SystemExit(f"atomzero instances already running: {ids}; destroy them first")

    instance_id = 0
    started = time.time()
    run_dir = Path(args.out_root) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    ok = False
    try:
        print(f"creating instance from offer {args.offer_id}...", flush=True)
        instance_id = create_instance(args, args.offer_id, key)
        print(f"instance id: {instance_id}", flush=True)
        inst = wait_running(instance_id, key)
        user, host, port = ssh_target(instance_id, key)
        print(f"ssh: {user}@{host}:{port}", flush=True)

        ssh = ssh_base(user, host, port)
        print("waiting for SSH...", flush=True)
        wait_ssh_ready(ssh)
        remote(ssh, "mkdir -p /workspace/out")
        print("uploading bundle...", flush=True)
        subprocess.run(
            [*scp_base(port), str(bundle), f"{user}@{host}:/workspace/atomzero-bundle.tar.gz"],
            check=True,
        )
        remote(
            ssh,
            "mkdir -p /workspace/atomzero && "
            "tar -xzf /workspace/atomzero-bundle.tar.gz -C /workspace/atomzero",
            timeout=300,
        )
        run_tag = args.run_tag or f"fullft_{args.layers}l"
        env_prefix = (
            f"LAYERS={args.layers} SEEDS='{args.seeds}' STEPS={args.steps} "
            f"BATCH={args.batch} RUN_TAG='{run_tag}' "
            f"TRAIN_DATA='{args.train_data}' CACHE_DIR='{args.cache_dir}' "
            f"COPY_CHECKPOINT={1 if args.copy_checkpoint else 0} "
        )
        # A plain nohup still leaks the SSH channel's extra fds, so the server
        # waits for the training process. Start a fully detached session with
        # close_fds=True instead; the command returns immediately.
        detach = (
            "python3 -c \"import subprocess; "
            "f=open('/workspace/job.out','wb'); "
            "subprocess.Popen(['bash','/workspace/atomzero/scripts/vastai/job.sh'], "
            "stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, "
            "start_new_session=True, close_fds=True); print('started')\""
        )
        start_cmd = f"cd /workspace && {env_prefix}{detach}"
        remote(ssh, start_cmd, timeout=60)

        # Confirm the detached process really exists before trusting it.
        time.sleep(8)
        check = remote(
            ssh,
            "pgrep -af '[v]astai/job.sh' | head -n 2; "
            "test -f /workspace/out/job.log && tail -n 1 /workspace/out/job.log || true",
            check=False,
        ).strip()
        if "job.sh" not in check and "===" not in check:
            log = remote(ssh, "tail -n 20 /workspace/job.out 2>/dev/null || true", check=False)
            raise SystemExit(f"job did not start.\n{check}\n{log[:800]}")
        print(f"job started: {check[:200]}", flush=True)
        print("polling...", flush=True)

        deadline = started + args.max_hours * 3600
        last_log = ""
        while time.time() < deadline:
            time.sleep(args.poll_seconds)
            state = remote(
                ssh,
                "if [ -f /workspace/out/DONE ]; then echo DONE; "
                "elif [ -f /workspace/out/FAILED ]; then echo FAILED; "
                "elif pgrep -f '[v]astai/job.sh' >/dev/null; then echo RUNNING; "
                "else echo CRASHED; fi; "
                "tail -n 2 /workspace/out/job.log 2>/dev/null | tr '\\n' ' '",
                check=False,
            )
            state = state.strip()
            if state != last_log:
                print(f"  {state[:300]}", flush=True)
                last_log = state
            first = state.splitlines()[0].strip() if state else ""
            if first == "DONE":
                ok = True
                break
            if first in ("FAILED", "CRASHED"):
                print(f"job stopped: {first}", flush=True)
                break

        print("downloading out/...", flush=True)
        subprocess.run(
            [*scp_base(port), "-r", f"{user}@{host}:/workspace/out", str(run_dir)],
            check=False,
        )
        print(f"results: {run_dir}")
        if not ok:
            print("job did not report DONE", flush=True)
    finally:
        elapsed_h = (time.time() - started) / 3600.0
        print(f"elapsed {elapsed_h:.2f} h, estimated compute ${estimate_cost(args.dph, elapsed_h):.2f}")
        if instance_id and not args.keep:
            destroy(instance_id, key)
        elif instance_id:
            print(f"WARNING: instance {instance_id} kept alive; destroy it manually", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
