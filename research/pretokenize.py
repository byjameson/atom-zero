"""Pretokenize a training JSONL into a compact memmap cache.

The full fine-tune control (:mod:`research.finetune_copy_control`) tokenizes
every row on the CPU inside the training loop. On a rented GPU this makes the
data loader the bottleneck: GPU utilization drops while Python formats rows,
runs the tokenizer, builds the assistant mask and (for ``content_weight > 1``)
decodes every scored token to decide whether it is alphanumeric.

This module moves that work out of the loop. It reuses the trainer's own
:func:`research.train_adapter_candidate._encode` and
:func:`research.train_adapter_candidate._token_weights`, so the cached tensors
have exactly the same ids, labels and token weights as the on-the-fly path.

    python research/pretokenize.py \
      --data data/copy_curriculum_v1/train.jsonl \
      --tokenizer "$env:USERPROFILE\\.cache\\atomzero\\atom-zero-49m\\chat_en_bpe8k.json" \
      --out data/pretok/copy_curriculum_v1 \
      --max-seq 256 --content-weight 5.0 --manifest

The cache is a directory with four flat little-endian ``.npy`` memmap files
(``input_ids`` int32, ``labels`` int32, ``weights`` float16, ``lengths``
int32) and a content-free ``meta.json``. Rows that produce no scorable label
(after :func:`_encode` truncation/masking) are skipped and counted; nothing
is padded because the trainer already encodes rows individually.

The cache is **training data only**. Never build it from an eval JSONL and
never pass an eval file as ``--data``. It stores no prompt or answer text, but
token ids are reversible in principle, so treat the directory as sensitive.
Rebuild the cache whenever the tokenizer, ``--max-seq`` or
``--content-weight`` changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

try:
    from research.adapter_runtime import load_tokenizer, read_rows
    from research.train_adapter_candidate import _encode, _token_weights
except ImportError:  # running as `python research/pretokenize.py`
    from adapter_runtime import load_tokenizer, read_rows
    from train_adapter_candidate import _encode, _token_weights

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "atomzero.pretokenized.v1"
PURPOSE = "training_only"
PAYLOAD_FILES = ("input_ids.npy", "labels.npy", "weights.npy", "lengths.npy")
DTYPE_NAMES = {
    "input_ids.npy": "int32",
    "labels.npy": "int32",
    "weights.npy": "float16",
    "lengths.npy": "int32",
}
_ARRAY_DTYPES = {
    "input_ids.npy": np.dtype("<i4"),
    "labels.npy": np.dtype("<i4"),
    "weights.npy": np.dtype("<f2"),
    "lengths.npy": np.dtype("<i4"),
}


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def _display_path(path: str | Path) -> str:
    value = Path(path)
    try:
        return value.resolve().relative_to(ROOT).as_posix()
    except (OSError, ValueError):
        return value.as_posix()


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(files: dict[str, dict]) -> str:
    """Aggregate hash over the per-file hashes in a fixed order."""
    digest = hashlib.sha256()
    for name in PAYLOAD_FILES:
        digest.update(f"{name}:{files[name]['sha256']}\n".encode("utf-8"))
    return digest.hexdigest()


def build_cache(rows: list[dict], tok, out_dir: str | Path, *,
                max_seq: int = 256, content_weight: float = 1.0,
                tokenizer_sha256: str = "",
                tokenizer_name: str = "",
                data_sha256: str = "",
                data_name: str = "") -> dict:
    """Write a pretokenized cache and return its metadata.

    Rows are encoded with the trainer helpers, so a cache round-trip is
    byte-for-byte equivalent to calling ``_encode`` and ``_token_weights``.
    Rows with fewer than three tokens or with no scored label are skipped and
    counted in ``skipped_rows`` / ``skip_reasons``. Raises :class:`ValueError`
    if no row is scorable, because an empty cache cannot train anything.
    """
    if max_seq < 3:
        raise ValueError("max_seq must be >= 3")
    if not math.isfinite(content_weight) or content_weight < 1.0:
        raise ValueError("content_weight must be finite and >= 1.0")
    if not tokenizer_sha256:
        to_str = getattr(tok, "to_str", None)
        if callable(to_str):
            tokenizer_sha256 = hashlib.sha256(
                to_str().encode("utf-8")).hexdigest()
            tokenizer_name = tokenizer_name or "tokenizer.to_str()"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    input_ids: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    lengths: list[int] = []
    manifest_rows: list[dict] = []
    skip_reasons = {"too_short": 0, "no_scorable_token": 0}
    for index, row in enumerate(rows):
        encoded = _encode(tok, row, max_seq, device)
        if encoded is None:
            skip_reasons["too_short"] += 1
            manifest_rows.append({"index": index, "kept": False,
                                  "reason": "too_short"})
            continue
        x, y = encoded
        token_weights = _token_weights(tok, y, content_weight)
        denominator = token_weights.sum()
        if float(denominator) == 0.0:
            skip_reasons["no_scorable_token"] += 1
            manifest_rows.append({"index": index, "kept": False,
                                  "reason": "no_scorable_token"})
            continue
        input_ids.append(
            np.asarray(x[0].tolist(), dtype=np.dtype("<i4")))
        labels.append(np.asarray(y[0].tolist(), dtype=np.dtype("<i4")))
        weights.append(
            np.asarray(token_weights[0].tolist(), dtype=np.dtype("<f2")))
        lengths.append(int(x.size(1)))
        manifest_rows.append({"index": index, "kept": True,
                              "tokens": int(x.size(1))})
    cached_rows = len(lengths)
    if cached_rows == 0:
        reasons = ", ".join(f"{name}={count}"
                            for name, count in skip_reasons.items())
        raise ValueError(
            f"no scorable rows in the input; skipped {len(rows)} rows "
            f"({reasons})")
    arrays = {
        "input_ids.npy": np.concatenate(input_ids).astype(
            np.dtype("<i4"), copy=False),
        "labels.npy": np.concatenate(labels).astype(
            np.dtype("<i4"), copy=False),
        "weights.npy": np.concatenate(weights).astype(
            np.dtype("<f2"), copy=False),
        "lengths.npy": np.asarray(lengths, dtype=np.dtype("<i4")),
    }
    files: dict[str, dict] = {}
    for name in PAYLOAD_FILES:
        path = out_dir / name
        np.save(path, arrays[name], allow_pickle=False)
        files[name] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
            "dtype": DTYPE_NAMES[name],
        }
    payload_sha256 = _payload_sha256(files)
    skipped_rows = sum(skip_reasons.values())
    tokens = int(sum(lengths))
    meta = {
        "schema": SCHEMA,
        "purpose": PURPOSE,
        "cached_rows": cached_rows,
        "skipped_rows": skipped_rows,
        "rows_seen": len(rows),
        "tokens": tokens,
        "max_seq": int(max_seq),
        "content_weight": float(content_weight),
        "skip_reasons": skip_reasons,
        "tokenizer": {
            "sha256": tokenizer_sha256,
            "file": tokenizer_name,
        },
        "data": {
            "sha256": data_sha256,
            "file": data_name,
            "rows": len(rows),
        },
        "dtypes": dict(DTYPE_NAMES),
        "files": files,
        "sha256": payload_sha256,
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8")
    meta["manifest_rows"] = manifest_rows
    return meta


def write_manifest(path: str | Path, meta: dict) -> Path:
    """Write a content-free per-row manifest (index, kept, tokens/reason)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": f"{SCHEMA}.manifest",
        "purpose": PURPOSE,
        "cached_rows": meta["cached_rows"],
        "skipped_rows": meta["skipped_rows"],
        "skip_reasons": meta["skip_reasons"],
        "rows": meta.get("manifest_rows", []),
    }
    text = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def _error_prefix(path: Path) -> str:
    return (f"invalid pretokenized cache at {path.as_posix()!r}; "
            "rebuild it with `python research/pretokenize.py`")


def _check_file_hashes(path: Path, meta: dict) -> None:
    files = meta.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"{_error_prefix(path)}: meta.json has no files map")
    for name in PAYLOAD_FILES:
        entry = files.get(name)
        if not isinstance(entry, dict) or "sha256" not in entry:
            raise ValueError(
                f"{_error_prefix(path)}: meta.json has no hash for {name}")
        payload = path / name
        if not payload.is_file():
            raise FileNotFoundError(
                f"{_error_prefix(path)}: payload file {name} is missing")
        actual = file_sha256(payload)
        if actual != entry["sha256"]:
            raise ValueError(
                f"{_error_prefix(path)}: payload hash mismatch for {name} "
                f"(metadata={entry['sha256'][:12]}, file={actual[:12]}); "
                "the cache is corrupt or was modified")
    expected = meta.get("sha256")
    actual = _payload_sha256(files)
    if expected != actual:
        raise ValueError(
            f"{_error_prefix(path)}: aggregate sha256 mismatch "
            f"(metadata={expected}, recomputed={actual})")


class PretokenizedCache:
    """Read-only memmap view over a pretokenized training cache."""

    def __init__(self, path: str | Path, meta: dict, arrays: dict):
        self.path = Path(path)
        self.meta = meta
        self._input_ids = arrays["input_ids.npy"]
        self._labels = arrays["labels.npy"]
        self._weights = arrays["weights.npy"]
        lengths = arrays["lengths.npy"]
        self._offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        if len(lengths):
            np.cumsum(lengths, out=self._offsets[1:])
        self._lengths = lengths

    def __len__(self) -> int:
        return int(self.meta["cached_rows"])

    def __getitem__(self, index: int):
        index = int(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(
                f"cache row {index} is outside 0..{len(self) - 1}")
        start = int(self._offsets[index])
        end = int(self._offsets[index + 1])
        x = torch.from_numpy(np.array(
            self._input_ids[start:end], dtype=np.int64)).unsqueeze(0)
        y = torch.from_numpy(np.array(
            self._labels[start:end], dtype=np.int64)).unsqueeze(0)
        weights = torch.from_numpy(np.array(
            self._weights[start:end], dtype=np.float32)).unsqueeze(0)
        return x, y, weights

    def lengths(self) -> list[int]:
        """Token count per cached row."""
        return [int(value) for value in self._lengths]


def load_cache(cache_dir: str | Path, *, verify: bool = True
               ) -> PretokenizedCache:
    """Load and validate a cache directory.

    Missing directories and payload files raise :class:`FileNotFoundError`
    with the rebuild command; a wrong schema, corrupt payload or hash mismatch
    raises :class:`ValueError`.
    """
    path = Path(cache_dir).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(
            f"pretokenized cache directory not found: {path.as_posix()!r}; "
            "build it with `python research/pretokenize.py --data ... "
            "--tokenizer ... --out ...`")
    meta_path = path / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"{_error_prefix(path)}: meta.json is missing")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{_error_prefix(path)}: meta.json is not valid JSON: "
            f"{exc}") from exc
    if not isinstance(meta, dict):
        raise ValueError(f"{_error_prefix(path)}: meta.json is not an object")
    schema = meta.get("schema")
    if schema != SCHEMA:
        raise ValueError(
            f"{_error_prefix(path)}: schema {schema!r} is not {SCHEMA!r}; "
            "rebuild the cache with the current research/pretokenize.py")
    cached_rows = meta.get("cached_rows")
    if not isinstance(cached_rows, int) or cached_rows < 1:
        raise ValueError(
            f"{_error_prefix(path)}: cached_rows must be a positive integer")
    if verify:
        _check_file_hashes(path, meta)
    arrays = {}
    for name in PAYLOAD_FILES:
        payload = path / name
        if not payload.is_file():
            raise FileNotFoundError(
                f"{_error_prefix(path)}: payload file {name} is missing")
        try:
            array = np.load(payload, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"{_error_prefix(path)}: cannot read {name}: {exc}") from exc
        expected = _ARRAY_DTYPES[name]
        if array.ndim != 1:
            raise ValueError(
                f"{_error_prefix(path)}: {name} must be one-dimensional, "
                f"got shape {array.shape}")
        if (array.dtype.kind != expected.kind
                or array.dtype.itemsize != expected.itemsize):
            raise ValueError(
                f"{_error_prefix(path)}: {name} has dtype {array.dtype}, "
                f"expected {DTYPE_NAMES[name]}")
        arrays[name] = array
    lengths = arrays["lengths.npy"]
    input_ids = arrays["input_ids.npy"]
    labels = arrays["labels.npy"]
    weights = arrays["weights.npy"]
    if len(lengths) != cached_rows:
        raise ValueError(
            f"{_error_prefix(path)}: lengths has {len(lengths)} rows but "
            f"meta.json says cached_rows={cached_rows}")
    if len(input_ids) != len(labels) or len(input_ids) != len(weights):
        raise ValueError(
            f"{_error_prefix(path)}: payload lengths disagree "
            f"(input_ids={len(input_ids)}, labels={len(labels)}, "
            f"weights={len(weights)})")
    total = int(np.asarray(lengths, dtype=np.int64).sum())
    if total != len(input_ids):
        raise ValueError(
            f"{_error_prefix(path)}: lengths sum to {total} tokens but "
            f"input_ids has {len(input_ids)}")
    if (np.asarray(lengths) < 1).any():
        raise ValueError(
            f"{_error_prefix(path)}: every cached row needs at least "
            "one token")
    return PretokenizedCache(path, meta, arrays)


def load_training_cache(cache_dir: str | Path, *, max_seq: int,
                        content_weight: float,
                        verify: bool = True) -> PretokenizedCache:
    """Load a cache and check it matches the trainer settings."""
    cache = load_cache(cache_dir, verify=verify)
    cached_seq = int(cache.meta["max_seq"])
    if cached_seq != int(max_seq):
        raise ValueError(
            f"cache max_seq={cached_seq} does not match --max-seq "
            f"{int(max_seq)}; rebuild the cache with --max-seq "
            f"{int(max_seq)}")
    cached_weight = float(cache.meta["content_weight"])
    if not math.isclose(cached_weight, float(content_weight),
                        rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"cache content_weight={cached_weight} does not match "
            f"--content-weight {float(content_weight)}; rebuild the cache "
            "with the same weighting")
    return cache


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Pretokenize a training JSONL for the full fine-tune "
                    "control (training data only)")
    parser.add_argument("--data", required=True,
                        help="training JSONL with user/assistant rows")
    parser.add_argument("--tokenizer", required=True,
                        help="tokenizer JSON (chat_en_bpe8k.json)")
    parser.add_argument("--out", required=True,
                        help="cache directory to write")
    parser.add_argument("--max-seq", type=int, default=256)
    parser.add_argument("--content-weight", type=float, default=1.0)
    parser.add_argument(
        "--manifest", nargs="?", const="manifest.json", default="",
        help="write a content-free per-row manifest; a bare flag writes "
             "manifest.json inside --out")
    args = parser.parse_args(argv)
    if args.max_seq < 3:
        parser.error("--max-seq must be >= 3")
    if not math.isfinite(args.content_weight) or args.content_weight < 1.0:
        parser.error("--content-weight must be finite and >= 1.0")
    data_path = _resolve(args.data)
    tokenizer_path = _resolve(args.tokenizer)
    out_dir = _resolve(args.out)
    if not data_path.is_file():
        parser.error(f"--data is not a file: {data_path}")
    if not tokenizer_path.is_file():
        parser.error(f"--tokenizer is not a file: {tokenizer_path}")
    rows = read_rows(data_path)
    tok = load_tokenizer(tokenizer_path)
    meta = build_cache(
        rows, tok, out_dir, max_seq=args.max_seq,
        content_weight=args.content_weight,
        tokenizer_sha256=file_sha256(tokenizer_path),
        tokenizer_name=Path(tokenizer_path).name,
        data_sha256=file_sha256(data_path),
        data_name=Path(data_path).name)
    if args.manifest:
        manifest = Path(args.manifest)
        if not manifest.is_absolute():
            manifest = out_dir / manifest
        write_manifest(manifest, meta)
    summary = {
        "cache": _display_path(out_dir),
        "cached_rows": meta["cached_rows"],
        "skipped_rows": meta["skipped_rows"],
        "skip_reasons": meta["skip_reasons"],
        "tokens": meta["tokens"],
        "max_seq": meta["max_seq"],
        "content_weight": meta["content_weight"],
        "sha256": meta["sha256"],
    }
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
