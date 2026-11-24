"""Deterministic replay sampling for adapter training.

Fine-tuning on a narrow distribution can overwrite older behaviour. A replay
mix keeps a bounded fraction of the training steps on rows from an older
distribution so the adapter does not forget it.

The sampler is deliberately small and deterministic:

    from research.replay_sampler import sample_steps, summarize_mix

    sequence = sample_steps(primary, replay, steps=300, replay_weight=0.25)
    mix = summarize_mix(sequence, primary, replay)
    # {'primary': 225, 'replay': 75}

The fraction of replay steps is computed up front and shuffled with
``random.Random(seed)``. With no replay rows (or ``replay_weight == 0``) the
primary rows are cycled in their original order, which preserves the historic
trainer behaviour. Input lists and rows are never mutated.
"""
from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

DEFAULT_REPLAY_WEIGHT = 0.25
DEFAULT_SEED = 13

__all__ = [
    "DEFAULT_REPLAY_WEIGHT",
    "DEFAULT_SEED",
    "sample_steps",
    "summarize_mix",
]


def _rows(value: Any, name: str, *, allow_empty: bool) -> list[Mapping]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
            value, Sequence):
        raise TypeError(f"{name} must be a sequence of rows")
    rows = list(value)
    if not rows and not allow_empty:
        raise ValueError(f"{name} must contain at least one row")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"{name}[{index}] must be a mapping")
    return rows


def _steps(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("steps must be an int")
    if value < 0:
        raise ValueError("steps must be >= 0")
    return value


def _weight(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("replay_weight must be a number")
    weight = float(value)
    if not math.isfinite(weight):
        raise ValueError("replay_weight must be finite")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("replay_weight must be between 0.0 and 1.0")
    return weight


def _seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("seed must be an int")
    return value


def sample_steps(primary_rows: Sequence[Mapping],
                 replay_rows: Sequence[Mapping] | None,
                 steps: int, *,
                 replay_weight: float = DEFAULT_REPLAY_WEIGHT,
                 seed: int = DEFAULT_SEED) -> list[Mapping]:
    """Return ``steps`` training rows with a bounded replay fraction.

    ``replay_weight`` is the fraction of the sequence drawn from
    ``replay_rows`` and must be in ``0.0..1.0``. The count is rounded to the
    nearest step, so very small runs can differ by one step from the requested
    fraction. Selection uses ``random.Random(seed)`` and is reproducible.

    An empty replay list, ``None`` or a zero weight returns primary rows only,
    cycled in input order (the legacy trainer behaviour). Rows are returned by
    reference; neither input list is modified.
    """
    primary = _rows(primary_rows, "primary_rows", allow_empty=False)
    replay = ([] if replay_rows is None
              else _rows(replay_rows, "replay_rows", allow_empty=True))
    steps = _steps(steps)
    weight = _weight(replay_weight)
    seed = _seed(seed)
    if steps == 0:
        return []
    if not replay or weight == 0.0:
        return [primary[index % len(primary)] for index in range(steps)]
    replay_steps = min(steps, int(steps * weight + 0.5))
    flags = [True] * replay_steps + [False] * (steps - replay_steps)
    rng = random.Random(seed)
    rng.shuffle(flags)
    return [rng.choice(replay if use_replay else primary)
            for use_replay in flags]


def summarize_mix(sampled_rows: Sequence[Mapping],
                  primary_rows: Sequence[Mapping],
                  replay_rows: Sequence[Mapping] | None = None
                  ) -> dict[str, int]:
    """Count sampled rows that came from each source.

    Rows are matched to their source by object identity, because
    :func:`sample_steps` returns references to the input rows rather than
    copies. Rows that appear in both sources are counted as primary. A row
    that belongs to neither source raises :class:`ValueError`.
    """
    sampled = _rows(sampled_rows, "sampled_rows", allow_empty=True)
    primary = _rows(primary_rows, "primary_rows", allow_empty=False)
    replay = ([] if replay_rows is None
              else _rows(replay_rows, "replay_rows", allow_empty=True))
    primary_ids = {id(row) for row in primary}
    replay_ids = {id(row) for row in replay}
    counts = {"primary": 0, "replay": 0}
    for row in sampled:
        if id(row) in primary_ids:
            counts["primary"] += 1
        elif id(row) in replay_ids:
            counts["replay"] += 1
        else:
            raise ValueError(
                "sampled row does not belong to primary_rows or replay_rows")
    return counts
