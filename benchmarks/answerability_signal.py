"""Measure an independent answerability signal and select a refusal policy.

This is an experiment, not a production gate. It trains a local logistic model
on SQuAD v2 train contexts, calibrates it with isotonic regression on a
held-out calibration split, and evaluates answer/refuse policy on the same
200-question SQuAD v2 validation sample used by the citation-gate benchmark.

The signal uses retrieval distribution, query coverage, question type,
answer-type candidates, entity/number alignment and retrieval confidence. It
does not use CitationGate probabilities.

Run (benchmark dependencies: datasets, scikit-learn, numpy):

    python benchmarks/answerability_signal.py \\
        --aisc-root "C:\...\ai-systems-algebra"

Artifacts are written to ``benchmarks/results/answerability_signal``:

- answerability_observations.csv
- answerability_profile.json
- answerability_report.md
- answerability.toml
- aisc_policy_*.txt / aisc_check.txt
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from atomzero.answerability import FEATURE_NAMES, extract_features, vectorize
from atomzero.doc_memory import DocMemory

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = "benchmarks/results/answerability_signal"
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
TARGETS = {
    "refusal_recall": 0.90,
    "refusal_precision": 0.90,
    "answerable_coverage": 0.80,
}
MIN_TRAIN = 30


def load_squad_rows(split: str, max_contexts: int, per_context: int,
                    seed: int, exclude_contexts: set[str] | None = None,
                    max_questions: int | None = None):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "install the benchmark dependency first: pip install datasets"
        ) from exc
    rows = load_dataset("rajpurkar/squad_v2", split=split).shuffle(seed=seed)
    contexts: dict[str, str] = {}
    selected = []
    counts: dict[str, int] = {}
    for row in rows:
        context = row["context"]
        if exclude_contexts and context in exclude_contexts:
            continue
        if context not in contexts:
            if len(contexts) >= max_contexts:
                continue
            contexts[context] = f"{split}-{len(contexts):03d}"
        limit = (max_questions if max_questions is not None
                 else max_contexts * per_context)
        if len(selected) >= limit:
            break
        if counts.get(context, 0) >= per_context:
            continue
        selected.append(row)
        counts[context] = counts.get(context, 0) + 1
    return selected, contexts


def extract_rows(rows, contexts, db_path: Path, label: str = "") -> list[dict]:
    docs = DocMemory(db_path)
    try:
        titles = {}
        for row in rows:
            titles.setdefault(row["context"], row.get("title", ""))
        for context, source in contexts.items():
            docs.add_text(source, context, title=titles.get(context, ""))
        output = []
        for index, row in enumerate(rows):
            hits = docs.search(row["question"], k=6, rerank=True)
            output.append({
                "question_id": row.get("id", f"{label}-{index}"),
                "question": row["question"],
                "context_text": " ".join(
                    str(hit.get("chunk", "")) for hit in hits[:3]),
                "features": extract_features(row["question"], hits),
                "label": int(bool(row["answers"]["text"])),
                "group": contexts[row["context"]],
                "split": label,
            })
        return output
    finally:
        docs.close()


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


def _raw(profile: dict, features: dict) -> float:
    return profile["intercept"] + sum(
        coefficient * features[name]
        for coefficient, name in zip(profile["coefficients"], FEATURE_NAMES))


def _calibrated(profile: dict, features: dict) -> float:
    value = _sigmoid(_raw(profile, features))
    x_values = profile["calibration"]["x"]
    y_values = profile["calibration"]["y"]
    if not x_values:
        return max(0.0, min(1.0, value))
    return max(0.0, min(1.0, float(
        __import__("numpy").interp(value, x_values, y_values))))


def brier_score(labels: list[int], probabilities: list[float]) -> float:
    if not labels:
        return 0.0
    return sum((probability - label) ** 2
               for probability, label in zip(probabilities, labels)) / len(labels)


def expected_calibration_error(labels: list[int], probabilities: list[float],
                               bins: int = 10) -> float:
    total = len(probabilities)
    if total == 0:
        return 0.0
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        members = [i for i, probability in enumerate(probabilities)
                   if (low <= probability < high
                       or (index == bins - 1 and probability == 1.0))]
        if not members:
            continue
        mean_prediction = sum(probabilities[i] for i in members) / len(members)
        accuracy = sum(labels[i] for i in members) / len(members)
        error += (len(members) / total) * abs(mean_prediction - accuracy)
    return error


def roc_auc(labels: list[int], probabilities: list[float]) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, probabilities))
    except (ImportError, ValueError):
        return 0.0


def wilson_interval(successes: int, total: int, z: float = 1.96):
    if total <= 0:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    spread = (z * math.sqrt(p * (1 - p) / total
                            + z * z / (4 * total * total)) / denominator)
    return [round(max(0.0, center - spread), 4),
            round(min(1.0, center + spread), 4)]


def policy_point(labels: list[int], probabilities: list[float],
                 threshold: float) -> dict:
    accepted = [i for i, probability in enumerate(probabilities)
                if probability >= threshold]
    accepted_set = set(accepted)
    answerable = [i for i, label in enumerate(labels) if label == 1]
    no_answer = [i for i, label in enumerate(labels) if label == 0]
    answerable_accepted = sum(1 for i in accepted if labels[i] == 1)
    no_answer_refused = sum(1 for i in no_answer if i not in accepted_set)
    refused = len(labels) - len(accepted)
    refusal_precision = (no_answer_refused / refused) if refused else 0.0
    refusal_recall = (no_answer_refused / len(no_answer)) if no_answer else 0.0
    coverage = ((answerable_accepted / len(answerable))
                if answerable else 0.0)
    accepted_correct = sum(1 for i in accepted if labels[i] == 1)
    return {
        "threshold": round(threshold, 4),
        "accepted": len(accepted),
        "refused": refused,
        "answerable": len(answerable),
        "no_answer": len(no_answer),
        "answerable_accepted": answerable_accepted,
        "no_answer_refused": no_answer_refused,
        "answerable_coverage": round(coverage, 4),
        "refusal_precision": round(refusal_precision, 4),
        "refusal_recall": round(refusal_recall, 4),
        "overall_accuracy": round(
            (accepted_correct + no_answer_refused) / max(1, len(labels)), 4),
        "refusal_precision_wilson_95": wilson_interval(
            no_answer_refused, refused),
        "refusal_recall_wilson_95": wilson_interval(
            no_answer_refused, len(no_answer)),
        "answerable_coverage_wilson_95": wilson_interval(
            answerable_accepted, len(answerable)),
    }


def policy_table(labels: list[int], probabilities: list[float]) -> list[dict]:
    return [policy_point(labels, probabilities, threshold)
            for threshold in THRESHOLDS]


def select_threshold(labels: list[int], probabilities: list[float]) -> dict:
    points = policy_table(labels, probabilities)
    feasible = [point for point in points
                if point["refusal_recall"] >= TARGETS["refusal_recall"]
                and point["refusal_precision"] >= TARGETS["refusal_precision"]
                and point["answerable_coverage"]
                >= TARGETS["answerable_coverage"]]
    if feasible:
        selected = max(feasible, key=lambda point:
                       (point["overall_accuracy"],
                        point["answerable_coverage"],
                        -abs(point["threshold"] - 0.5)))
        return {"feasible": True, "point": selected, "misses": []}
    selected = min(points, key=lambda point: (
        max(0.0, TARGETS["refusal_recall"] - point["refusal_recall"])
        + max(0.0, TARGETS["refusal_precision"]
              - point["refusal_precision"])
        + max(0.0, TARGETS["answerable_coverage"]
              - point["answerable_coverage"]),
        abs(point["threshold"] - 0.5)))
    misses = []
    for key, target in TARGETS.items():
        if selected[key] < target:
            misses.append(f"{key} {selected[key]:.3f} < {target:.2f}")
    return {"feasible": False, "point": selected, "misses": misses}


def fit_logistic(train: list[dict], calibration: list[dict]) -> dict:
    try:
        import numpy as np
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SystemExit(
            "install the benchmark dependencies first: "
            "pip install datasets scikit-learn"
        ) from exc
    x_train = np.array([vectorize(row["features"]) for row in train])
    y_train = np.array([row["label"] for row in train])
    x_cal = np.array([vectorize(row["features"]) for row in calibration])
    y_cal = np.array([row["label"] for row in calibration])
    best = None
    for c_value in (0.03, 0.1, 0.3, 1.0, 3.0):
        scaler = StandardScaler().fit(x_train)
        model = LogisticRegression(
            C=c_value, max_iter=3000, class_weight="balanced",
            random_state=0)
        model.fit(scaler.transform(x_train), y_train)
        raw_cal = 1.0 / (1.0 + np.exp(-model.decision_function(
            scaler.transform(x_cal))))
        brier = float(np.mean((raw_cal - y_cal) ** 2))
        ece = expected_calibration_error(
            y_cal.tolist(), raw_cal.tolist())
        candidate = {
            "C": c_value,
            "scaler": scaler,
            "model": model,
            "raw_cal": raw_cal,
            "brier": brier,
            "ece": ece,
        }
        if best is None or (brier, ece) < (best["brier"], best["ece"]):
            best = candidate
    isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    isotonic.fit(best["raw_cal"], y_cal)
    grid = np.unique(np.concatenate(([0.0, 1.0], best["raw_cal"])))
    calibrated = isotonic.predict(grid)
    coefficients = best["model"].coef_[0] / best["scaler"].scale_
    intercept = (best["model"].intercept_[0]
                 - float(np.dot(coefficients, best["scaler"].mean_)))
    profile = {
        "features": list(FEATURE_NAMES),
        "coefficients": [float(value) for value in coefficients],
        "intercept": float(intercept),
        "calibration": {
            "x": [float(value) for value in grid],
            "y": [float(value) for value in calibrated],
        },
        "model": {
            "type": "logistic",
            "C": best["C"],
            "scaler_mean": [float(value) for value in best["scaler"].mean_],
            "scaler_scale": [float(value) for value in best["scaler"].scale_],
        },
        "calibration_brier_raw": round(best["brier"], 6),
        "calibration_ece_raw": round(best["ece"], 6),
    }
    return profile


def predict(rows: list[dict], profile: dict) -> list[float]:
    return [_calibrated(profile, row["features"]) for row in rows]


def write_observations(path: Path, rows: list[dict],
                       probabilities: list[float], split: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["question_id", "prediction", "outcome", "answerable", "split"])
        for row, probability in zip(rows, probabilities):
            writer.writerow([
                row["question_id"], f"{probability:.6f}", row["label"],
                row["label"], split])


def write_toml(path: Path, profile: dict, test_point: dict) -> None:
    lines = [
        'name = "AtomZeroAnswerability"',
        'version = "0.1"',
        'description = "Independent answerability signal experiment"',
        "min_assurance = 0.80",
        "max_context_tokens = 50000",
        f"answerability_n = {test_point['answerable'] + test_point['no_answer']}",
        f"answerability_brier = {profile['metrics']['brier']:.6f}",
        f"answerability_ece = {profile['metrics']['ece']:.6f}",
        f"refusal_precision = {test_point['refusal_precision']:.6f}",
        f"refusal_recall = {test_point['refusal_recall']:.6f}",
        f"answerable_coverage = {test_point['answerable_coverage']:.6f}",
        f"refusal_threshold = {test_point['threshold']:.6f}",
        "min_refusal_precision = 0.90",
        "min_refusal_recall = 0.90",
        "min_answerable_coverage = 0.80",
        "",
        "[[components]]",
        'name = "AnswerabilitySignal"',
        'input_type = "Query"',
        'output_type = "Decision"',
        'kind = "model"',
        "assurance = 0.80",
        "verified = true",
        'evidence = "held-out SQuAD v2 answerability experiment"',
        'guarantees = ["calibrated answerability probability"]',
        "context_tokens = 2000",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_aisc_policy(aisc_root: str, csv_path: Path,
                    output_path: Path) -> dict:
    cmd = [
        sys.executable, "-m", "aisc", "policy", str(csv_path),
        "--thresholds", ",".join(str(value) for value in THRESHOLDS),
        "--min-refusal-recall", str(TARGETS["refusal_recall"]),
        "--min-refusal-precision", str(TARGETS["refusal_precision"]),
        "--min-answerable-coverage", str(TARGETS["answerable_coverage"]),
        "--json",
    ]
    completed = subprocess.run(
        cmd, cwd=aisc_root, capture_output=True, text=True, encoding="utf-8")
    output_path.write_text(
        "$ " + " ".join(cmd) + "\n\n"
        + (completed.stdout or "") + (completed.stderr or ""),
        encoding="utf-8")
    if completed.returncode != 0:
        return {"error": (completed.stderr or completed.stdout).strip()}
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {"error": f"aisc policy JSON error: {exc}"}


def run_aisc_check(aisc_root: str, toml_path: Path,
                   output_path: Path) -> dict:
    cmd = [
        sys.executable, "-m", "aisc", "check", str(toml_path),
        "--require-calibration", "--strict",
    ]
    completed = subprocess.run(
        cmd, cwd=aisc_root, capture_output=True, text=True, encoding="utf-8")
    output_path.write_text(
        "$ " + " ".join(cmd) + "\n\n"
        + (completed.stdout or "") + (completed.stderr or ""),
        encoding="utf-8")
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _metric_summary(labels: list[int], probabilities: list[float]) -> dict:
    return {
        "n": len(labels),
        "answerable": sum(labels),
        "no_answer": len(labels) - sum(labels),
        "brier": round(brier_score(labels, probabilities), 6),
        "ece": round(expected_calibration_error(labels, probabilities), 6),
        "auc": round(roc_auc(labels, probabilities), 6),
    }


def write_report(path: Path, payload: dict) -> None:
    lines = [
        "# Answerability signal and refusal policy",
        "",
        f"Commit: `{payload['commit']}`",
        f"Dataset / split: {payload['dataset']}",
        f"Signal features: {len(FEATURE_NAMES)} local retrieval and "
        "answer-type features; CitationGate probabilities are not used.",
        "Calibration method: logistic model + isotonic calibration on a "
        "context-grouped calibration split.",
        f"n: train={payload['train_n']}, calibration={payload['calibration_n']}, "
        f"held-out test={payload['test']['n']}",
        f"Brier / ECE: {payload['test']['brier']:.4f} / "
        f"{payload['test']['ece']:.4f}",
        f"Threshold: {payload['threshold']:.2f}",
        f"Refusal precision / recall: "
        f"{payload['test_point']['refusal_precision']:.4f} / "
        f"{payload['test_point']['refusal_recall']:.4f}",
        f"Answerable coverage: "
        f"{payload['test_point']['answerable_coverage']:.4f}",
        "",
        "## Held-out results",
        "",
        "| metric | value | target | pass |",
        "|---|---:|---:|:---:|",
    ]
    for name, target in TARGETS.items():
        value = payload["test_point"][name]
        lines.append(
            f"| {name} | {value:.4f} | {target:.2f} | "
            f"{'yes' if value >= target else 'no'} |")
    lines.extend([
        f"| answerability Brier | {payload['test']['brier']:.4f} | 0.15 | "
        f"{'yes' if payload['test']['brier'] <= 0.15 else 'no'} |",
        f"| answerability ECE | {payload['test']['ece']:.4f} | 0.10 | "
        f"{'yes' if payload['test']['ece'] <= 0.10 else 'no'} |",
        "",
        "Wilson 95% intervals:",
        "",
        f"- refusal precision: "
        f"{payload['test_point']['refusal_precision_wilson_95']}",
        f"- refusal recall: "
        f"{payload['test_point']['refusal_recall_wilson_95']}",
        f"- answerable coverage: "
        f"{payload['test_point']['answerable_coverage_wilson_95']}",
        "",
        "## Threshold table (held-out)",
        "",
        "| threshold | answerable cov. | refusal precision | refusal recall | "
        "overall |",
        "|---:|---:|---:|---:|---:|",
    ])
    for point in payload["test_table"]:
        lines.append(
            f"| {point['threshold']:.1f} | "
            f"{point['answerable_coverage']:.3f} | "
            f"{point['refusal_precision']:.3f} | "
            f"{point['refusal_recall']:.3f} | "
            f"{point['overall_accuracy']:.3f} |")
    lines.extend([
        "",
        "## AISC policy",
        "",
        "```text",
        payload.get("aisc_policy_text", "").strip(),
        "```",
        "",
        "## AISC strict check",
        "",
        "```text",
        payload.get("aisc_check_text", "").strip(),
        "```",
        "",
        "## Decision and limitations",
        "",
    ])
    for item in payload["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--train-contexts", type=int, default=240)
    parser.add_argument("--cal-contexts", type=int, default=60)
    parser.add_argument("--test-contexts", type=int, default=50)
    parser.add_argument("--per-context", type=int, default=5)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--test-seed", type=int, default=0)
    parser.add_argument("--aisc-root", default="")
    parser.add_argument("--skip-aisc", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args(argv)
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("loading train contexts...", flush=True)
    train_rows, train_contexts = load_squad_rows(
        "train", args.train_contexts, args.per_context, args.seed)
    print("loading calibration contexts...", flush=True)
    cal_rows, cal_contexts = load_squad_rows(
        "train", args.cal_contexts, args.per_context, args.seed + 1,
        exclude_contexts=set(train_contexts))
    print("loading held-out test contexts...", flush=True)
    test_rows, test_contexts = load_squad_rows(
        "validation", args.test_contexts, args.per_context, args.test_seed,
        max_questions=200)

    with tempfile.TemporaryDirectory(prefix="az-answerability-") as folder:
        print(f"extracting {len(train_rows)} train features...", flush=True)
        train = extract_rows(
            train_rows, train_contexts, Path(folder) / "train.db", "train")
        print(f"extracting {len(cal_rows)} calibration features...", flush=True)
        calibration = extract_rows(
            cal_rows, cal_contexts, Path(folder) / "cal.db", "calibration")
        print(f"extracting {len(test_rows)} test features...", flush=True)
        test = extract_rows(
            test_rows, test_contexts, Path(folder) / "test.db", "heldout")

    if len(train) < MIN_TRAIN or len(calibration) < MIN_TRAIN:
        raise SystemExit("not enough train/calibration rows")

    print("fitting logistic model and isotonic calibration...", flush=True)
    profile = fit_logistic(train, calibration)
    cal_labels = [row["label"] for row in calibration]
    cal_probabilities = predict(calibration, profile)
    test_labels = [row["label"] for row in test]
    test_probabilities = predict(test, profile)

    selection = select_threshold(cal_labels, cal_probabilities)
    calibration_csv = out / "answerability_calibration.csv"
    write_observations(
        calibration_csv, calibration, cal_probabilities, "calibration")
    aisc_calibration = {}
    aisc_threshold = None
    if args.aisc_root and not args.skip_aisc:
        print("running AISC calibration policy...", flush=True)
        aisc_calibration = run_aisc_policy(
            args.aisc_root, calibration_csv,
            out / "aisc_policy_calibration.txt")
        recommendation = aisc_calibration.get("recommendation") or {}
        value = recommendation.get("threshold")
        if isinstance(value, (int, float)):
            aisc_threshold = float(value)
    if args.threshold is not None:
        threshold = args.threshold
    elif aisc_threshold is not None:
        threshold = aisc_threshold
    else:
        threshold = selection["point"]["threshold"]
    test_point = policy_point(test_labels, test_probabilities, threshold)
    test_table = policy_table(test_labels, test_probabilities)
    test_metrics = _metric_summary(test_labels, test_probabilities)
    cal_metrics = _metric_summary(cal_labels, cal_probabilities)
    profile["metrics"] = test_metrics
    profile["threshold"] = threshold
    profile["calibration_metrics"] = cal_metrics
    profile.update({
        "n": len(test),
        "brier": test_metrics["brier"],
        "ece": test_metrics["ece"],
        "refusal_precision": test_point["refusal_precision"],
        "refusal_recall": test_point["refusal_recall"],
        "answerable_coverage": test_point["answerable_coverage"],
        "wilson": {
            "refusal_precision": test_point["refusal_precision_wilson_95"],
            "refusal_recall": test_point["refusal_recall_wilson_95"],
            "answerable_coverage":
                test_point["answerable_coverage_wilson_95"],
        },
    })
    profile["selection"] = {
        "feasible_on_calibration": selection["feasible"],
        "threshold": threshold,
        "misses": selection["misses"],
        "source": ("manual" if args.threshold is not None
                   else "aisc" if aisc_threshold is not None else "internal"),
        "internal_threshold": selection["point"]["threshold"],
        "aisc_threshold": aisc_threshold,
    }
    profile["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    profile["dataset"] = "rajpurkar/squad_v2"
    profile["split"] = {
        "train": "train contexts",
        "calibration": "train contexts, disjoint from training",
        "heldout": "validation contexts (50 contexts / 200 questions)",
    }
    profile["n"] = {
        "train": len(train),
        "calibration": len(calibration),
        "heldout": len(test),
    }
    import subprocess as _subprocess
    try:
        commit = _subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=_subprocess.DEVNULL).strip()
    except (OSError, _subprocess.SubprocessError):
        commit = ""
    profile["commit"] = commit
    profile["dataset_hash"] = hashlib.sha256(
        json.dumps([row["question_id"] for row in test], sort_keys=True)
        .encode("utf-8")).hexdigest()
    profile["calibration_date"] = time.strftime("%Y-%m-%d")
    profile["recalibration_due"] = time.strftime(
        "%Y-%m-%d", time.localtime(time.time() + 90 * 86400))

    observations = out / "answerability_observations.csv"
    write_observations(observations, test, test_probabilities, "heldout")
    profile_path = out / "answerability_profile.json"
    profile_path.write_text(
        json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    toml_path = out / "answerability.toml"
    write_toml(toml_path, profile, test_point)

    aisc_policy_text = ""
    aisc_check_text = ""
    if args.aisc_root and not args.skip_aisc:
        print("running AISC held-out policy/check...", flush=True)
        policy = run_aisc_policy(
            args.aisc_root, observations, out / "aisc_policy_heldout.txt")
        check = run_aisc_check(
            args.aisc_root, toml_path, out / "aisc_check.txt")
        aisc_policy_text = (out / "aisc_policy_heldout.txt").read_text(
            encoding="utf-8")
        aisc_check_text = (out / "aisc_check.txt").read_text(
            encoding="utf-8")
        profile["aisc"] = {
            "calibration_policy": aisc_calibration,
            "policy": policy,
            "check_returncode": check["returncode"],
            "check_stdout": check["stdout"],
            "check_stderr": check["stderr"],
        }
        profile_path.write_text(
            json.dumps(profile, indent=2) + "\n", encoding="utf-8")

    limitations = [
        "SQuAD v2 no-answer questions share entities and vocabulary with the "
        "source paragraph; this is a stress distribution, not a production "
        "pilot.",
        "The signal is retrieval-feature based; no NLI or QA model is used.",
        "The 0.90 refusal target is not met; CitationGate remains the only "
        "default refusal gate.",
        "The answerability gate is implemented behind an environment flag "
        "and defaults to off, so this experiment does not change product "
        "behavior.",
        "No 0.98 assurance claim is made.",
    ]
    payload = {
        "commit": commit,
        "dataset": "rajpurkar/squad_v2 validation (50 contexts / 200 questions)",
        "train_n": len(train),
        "calibration_n": len(calibration),
        "calibration": cal_metrics,
        "test": test_metrics,
        "threshold": threshold,
        "test_point": test_point,
        "test_table": test_table,
        "selection": selection,
        "aisc_policy_text": aisc_policy_text,
        "aisc_check_text": aisc_check_text,
        "limitations": limitations,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    write_report(out / "answerability_report.md", payload)
    print(json.dumps({
        "output": str(out),
        "train_n": len(train),
        "calibration_n": len(calibration),
        "test_n": len(test),
        "threshold": threshold,
        "brier": test_metrics["brier"],
        "ece": test_metrics["ece"],
        "refusal_precision": test_point["refusal_precision"],
        "refusal_recall": test_point["refusal_recall"],
        "answerable_coverage": test_point["answerable_coverage"],
        "selection_feasible": selection["feasible"],
        "selection_misses": selection["misses"],
        "elapsed_seconds": payload["elapsed_seconds"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
