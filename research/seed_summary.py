"""Summarize the seed spread of product-suite reports.

    python research/seed_summary.py \
        --inputs research/results/product_suite_j_expanded.json \
                 research/results/product_suite_j_s72.json \
                 research/results/product_suite_j_s73.json \
        --out research/results/seed_summary_j.json

The report is content-free: aggregate rates, run names and summary statistics
only. It does not read prompts, answers or eval data.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "benign_useful_rate",
    "benign_gate_rate",
    "single_turn_useful_rate",
    "multi_turn_useful_rate",
    "multi_turn_both_turns_useful_rate",
    "copy_exact",
    "content_loss_ratio",
    "safety_pass_rate",
    "rag_pass_rate",
)


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else ROOT / value


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _run_name(report: Mapping[str, Any], path: Path) -> str:
    checkpoints = report.get("checkpoints") or {}
    name = checkpoints.get("candidate") or path.stem
    return str(name)


def _metric_values(report: Mapping[str, Any], key: str) -> float | None:
    value = _number(report.get(key))
    if value is not None:
        return value
    # Older expanded-suite reports predate benign_gate_rate; the expanded
    # gate uses the single-turn rate, so fall back to that before the overall
    # historical rate.
    if key == "benign_gate_rate":
        return (_number(report.get("single_turn_useful_rate"))
                or _number(report.get("benign_useful_rate")))
    return None


def summarize(reports: Sequence[Mapping[str, Any]]) -> dict:
    """Return per-run values and mean/min/max for each metric."""
    if not reports:
        raise ValueError("at least one report is required")
    runs = []
    for report in reports:
        method = report.get("method")
        if method and method != "product_suite_v1":
            raise ValueError(f"unexpected report method: {method!r}")
        run = {
            "run": _run_name(report, Path(str(report.get("_path", "")))),
            "status": report.get("status"),
            "product_ok": report.get("product_ok"),
        }
        for metric in METRICS:
            run[metric] = _metric_values(report, metric)
        runs.append(run)
    summary = {"runs": runs, "metrics": {}}
    for metric in METRICS:
        values = [run[metric] for run in runs if run[metric] is not None]
        if not values:
            summary["metrics"][metric] = None
            continue
        summary["metrics"][metric] = {
            "mean": round(statistics.fmean(values), 6),
            "min": round(min(values), 6),
            "max": round(max(values), 6),
            "n": len(values),
        }
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    reports = []
    for raw in args.inputs:
        path = _resolve(raw)
        report = json.loads(path.read_text(encoding="utf-8"))
        report["_path"] = str(path)
        reports.append(report)
    summary = summarize(reports)
    text = json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False)
    if args.out:
        out = _resolve(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
