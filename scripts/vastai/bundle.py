"""Build a reproducible Vast.ai experiment bundle.

The bundle contains the repository's tracked code plus the exact data files a
GPU run needs. Checkpoints are never bundled: the instance downloads the public
49M base from Hugging Face and writes new checkpoints locally.

Usage
-----
    python scripts/vastai/bundle.py
    python scripts/vastai/bundle.py --out dist/vastai/atomzero-bundle.tar.gz
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DATA = (
    "data/multi_short_v1/train_all_plus_distill.jsonl",
    "data/multi_short_v1/stage_2short/eval.jsonl",
    "data/multi_short_v1/stage_3short/eval.jsonl",
    "data/multi_short_v1/stage_4short/eval.jsonl",
    "data/staged_copy_v1/mixed_eval.jsonl",
    "data/staged_copy_v1/rehearsal_focus.jsonl",
    "data/eval_frozen/heldout_city_yaml.jsonl",
    "data/eval_frozen/heldout_city_river.jsonl",
    "data/fresh4_v1/eval.jsonl",
    "data/mixed_replay_v1/train.jsonl",
    "data/pretok/mixed_v1/input_ids.npy",
    "data/pretok/mixed_v1/labels.npy",
    "data/pretok/mixed_v1/weights.npy",
    "data/pretok/mixed_v1/lengths.npy",
    "data/pretok/mixed_v1/meta.json",
    "data/mixed_benign_f/train.jsonl",
    "data/pretok/mixed_benign_f/input_ids.npy",
    "data/pretok/mixed_benign_f/labels.npy",
    "data/pretok/mixed_benign_f/weights.npy",
    "data/pretok/mixed_benign_f/lengths.npy",
    "data/pretok/mixed_benign_f/meta.json",
    "data/benign_v1/prompts.jsonl",
    "data/selfharm_replay_v1/eval.jsonl",
)

EXCLUDE_PREFIXES = (".git/", "checkpoints/", "dist/", "data/", "__pycache__/")


def _git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    )
    return proc.stdout


def code_files(root: Path) -> list[str]:
    """Tracked and new-not-ignored files, excluding bulky local artifacts."""
    raw = _git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    names = [n for n in raw.split("\0") if n]
    kept = []
    for name in sorted(set(names)):
        if any(name.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
            continue
        if not (root / name).is_file():
            continue
        kept.append(name)
    return kept


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(data))


def build_bundle(root: Path, out: Path, data_files: tuple[str, ...]) -> dict:
    root = root.resolve()
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    missing = [name for name in data_files if not (root / name).is_file()]
    if missing:
        raise SystemExit("missing data files:\n  " + "\n  ".join(missing))

    names = code_files(root)
    entries: list[dict] = []
    with tarfile.open(out, "w:gz", compresslevel=9) as tar:
        for name in names:
            path = root / name
            data = path.read_bytes()
            _add_bytes(tar, name, data)
            entries.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        for name in data_files:
            path = root / name
            data = path.read_bytes()
            _add_bytes(tar, name, data)
            entries.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})

        manifest = {
            "schema": "atomzero.vastai_bundle.v1",
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_head": _git(root, "rev-parse", "HEAD").strip(),
            "git_dirty": bool(_git(root, "status", "--porcelain").strip()),
            "file_count": len(entries),
            "files": entries,
        }
        raw = json.dumps(manifest, indent=2).encode("utf-8")
        _add_bytes(tar, "VASTAI_MANIFEST.json", raw)

    return {
        "bundle": str(out),
        "bytes": out.stat().st_size,
        "sha256": _sha256(out),
        "file_count": len(entries),
        "data_files": list(data_files),
        "git_head": _git(root, "rev-parse", "--short", "HEAD").strip(),
        "git_dirty": bool(_git(root, "status", "--porcelain").strip()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Vast.ai experiment bundle")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="dist/vastai/atomzero-bundle.tar.gz")
    parser.add_argument(
        "--data", nargs="*", default=list(DEFAULT_DATA),
        help="data files to include (paths relative to the repo root)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    info = build_bundle(Path(args.root), Path(args.out), tuple(args.data))
    sidecar = Path(str(args.out) + ".json")
    sidecar.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print(f"bundle:  {info['bundle']}")
        print(f"size:    {info['bytes'] / (1024 * 1024):.2f} MB")
        print(f"files:   {info['file_count']}")
        print(f"sha256:  {info['sha256']}")
        print(f"head:    {info['git_head']} dirty={info['git_dirty']}")
        print(f"sidecar: {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
