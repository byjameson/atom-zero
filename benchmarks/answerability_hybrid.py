"""Hybrid answerability signal: retrieval features + tiny local QA model.

This is the strongest local experiment in the answerability batch. It combines
the retrieval/answer-type features from ``answerability_signal.py`` with the
no-answer/span-margin confidence of a tiny SQuAD v2 QA model
(``mrm8488/bert-tiny-finetuned-squadv2``), fits a logistic model, calibrates it
with prior-weighted isotonic regression on a separate calibration split, and
evaluates the same 200-question validation sample.

The hybrid profile is an experiment artifact. The shipped optional gate is the
feature-only profile in ``benchmarks/results/answerability_signal``; both
default to off.

Run:

    python benchmarks/answerability_hybrid.py \\
        --aisc-root "C:\...\ai-systems-algebra"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from atomzero.answerability import FEATURE_NAMES

try:
    from benchmarks.answerability_signal import (
        CALIBRATION_METHOD, DEFAULT_TARGET_PREVALENCE, TARGETS, THRESHOLDS,
        _metric_summary, extract_rows, isotonic_curve, load_squad_rows,
        out_of_fold_metrics, policy_point, policy_table,
        prior_weighted_isotonic, run_aisc_check, run_aisc_policy,
        select_threshold, write_observations, write_toml)
except ImportError:  # running as `python benchmarks/answerability_hybrid.py`
    from answerability_signal import (
        CALIBRATION_METHOD, DEFAULT_TARGET_PREVALENCE, TARGETS, THRESHOLDS,
        _metric_summary, extract_rows, isotonic_curve, load_squad_rows,
        out_of_fold_metrics, policy_point, policy_table,
        prior_weighted_isotonic, run_aisc_check, run_aisc_policy,
        select_threshold, write_observations, write_toml)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = "benchmarks/results/answerability_hybrid"
QA_MODEL = "mrm8488/bert-tiny-finetuned-squadv2"
MAX_LENGTH = 384
MAX_ANSWER = 30
QA_FEATURES = ("qa_no_answer", "qa_best", "qa_margin", "qa_span_len",
               "qa_prob")


def qa_features(model, tok, question: str, context: str,
                device) -> dict:
    encoding = tok(question, context, truncation=True, max_length=MAX_LENGTH,
                   return_tensors="pt")
    sequence = encoding.sequence_ids(0)
    encoded = {key: value.to(device) for key, value in encoding.items()}
    with torch.no_grad():
        output = model(**encoded)
    start = output.start_logits[0].float().cpu().numpy()
    end = output.end_logits[0].float().cpu().numpy()
    no_answer = float(start[0] + end[0])
    positions = [i for i, value in enumerate(sequence) if value == 1]
    if not positions:
        return {"qa_no_answer": no_answer, "qa_best": 0.0,
                "qa_margin": -no_answer, "qa_span_len": 0.0, "qa_prob": 0.0}
    start_values = start[positions]
    end_values = end[positions]
    best = -1e9
    best_len = 0
    for offset in range(len(start_values)):
        window = end_values[offset:offset + MAX_ANSWER]
        if not len(window):
            continue
        index = int(np.argmax(window))
        score = float(start_values[offset] + window[index])
        if score > best:
            best = score
            best_len = index + 1
    margin = best - no_answer
    top = max(no_answer, best)
    probability = float(np.exp(best - top)
                        / (np.exp(no_answer - top) + np.exp(best - top)))
    return {
        "qa_no_answer": no_answer,
        "qa_best": best,
        "qa_margin": margin,
        "qa_span_len": float(best_len),
        "qa_prob": probability,
    }


def _vector(row: dict) -> list[float]:
    return (list(row["features"].values())
            + [row["qa"][name] for name in QA_FEATURES])


def fit_hybrid(train: list[dict], calibration: list[dict],
               target_prevalence: float | None = None) -> dict:
    if target_prevalence is None:
        target_prevalence = DEFAULT_TARGET_PREVALENCE
    x_train = np.array([_vector(row) for row in train])
    y_train = np.array([row["label"] for row in train])
    x_cal = np.array([_vector(row) for row in calibration])
    y_cal = np.array([row["label"] for row in calibration])
    best = None
    for c_value in (0.03, 0.1, 0.3, 1.0, 3.0):
        scaler = StandardScaler().fit(x_train)
        model = LogisticRegression(
            C=c_value, max_iter=3000, class_weight="balanced",
            random_state=0).fit(scaler.transform(x_train), y_train)
        raw_cal = 1.0 / (1.0 + np.exp(-model.decision_function(
            scaler.transform(x_cal))))
        brier = float(np.mean((raw_cal - y_cal) ** 2))
        if best is None or brier < best["brier"]:
            best = {"brier": brier, "C": c_value, "scaler": scaler,
                    "model": model, "raw_cal": raw_cal}
    observed_prevalence = float(np.mean(y_cal))
    isotonic = prior_weighted_isotonic(
        best["raw_cal"].tolist(), y_cal.tolist(), target_prevalence)
    grid_values, calibrated = isotonic_curve(isotonic, best["raw_cal"])
    coefficients = best["model"].coef_[0] / best["scaler"].scale_
    intercept = (best["model"].intercept_[0]
                 - float(np.dot(coefficients, best["scaler"].mean_)))
    names = list(FEATURE_NAMES) + list(QA_FEATURES)
    return {
        "feature_names": names,
        "coefficients": [float(value) for value in coefficients],
        "intercept": float(intercept),
        "calibration": {
            "x": [float(value) for value in grid_values],
            "y": [float(value) for value in calibrated],
        },
        "calibration_method": CALIBRATION_METHOD,
        "prevalence": {
            "calibration": round(observed_prevalence, 6),
            "target": round(float(target_prevalence), 6),
        },
        "model": {
            "type": "hybrid-logistic",
            "C": best["C"],
            "qa_model": QA_MODEL,
            "max_length": MAX_LENGTH,
            "max_answer": MAX_ANSWER,
        },
        "calibration_brier_raw": round(best["brier"], 6),
    }


def raw_probabilities(rows: list[dict], profile: dict) -> list[float]:
    output = []
    for row in rows:
        vector = _vector(row)
        raw = profile["intercept"] + sum(
            coefficient * value
            for coefficient, value in zip(profile["coefficients"], vector))
        output.append(float(1.0 / (1.0 + np.exp(-raw))))
    return output


def predict(rows: list[dict], profile: dict) -> list[float]:
    return [
        max(0.0, min(1.0, float(np.interp(
            probability, profile["calibration"]["x"],
            profile["calibration"]["y"]))))
        for probability in raw_probabilities(rows, profile)
    ]


def write_report(path: Path, payload: dict) -> None:
    lines = [
        "# Hybrid answerability signal and refusal policy",
        "",
        f"Commit: `{payload['commit']}`",
        f"Dataset / split: {payload['dataset']}",
        "Signal features: 42 retrieval/answer-type features plus 5 tiny "
        "SQuAD v2 QA features (no-answer score, best span, margin, span "
        "length, answer probability). CitationGate probabilities are not used.",
        f"Calibration method: {payload['calibration_method']} "
        "(prior-weighted isotonic map fitted on the context-grouped "
        "calibration split).",
        f"Prevalence: calibration={payload['prevalence']['calibration']:.4f}, "
        f"target={payload['prevalence']['target']:.4f}",
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
        "## Calibration report",
        "",
        "| report | split | n | Brier | ECE | AUC |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for label, key in (
            ("in-sample", "calibration_in_sample"),
            ("out-of-fold", "calibration_oof")):
        report = payload[key]
        lines.append(
            f"| {label} | {report['split']} | {report['n']} | "
            f"{report['brier']:.4f} | {report['ece']:.4f} | "
            f"{report['auc']:.4f} |")
    lines.extend([
        "",
        "`calibration_metrics` in the profile is the out-of-fold report. The "
        "in-sample row is optimistic because the production map saw those "
        "rows.",
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
        "## Baselines",
        "",
        "| signal | held-out AUC | Brier | ECE |",
        "|---|---:|---:|---:|",
        f"| feature-only logistic | {payload['baselines']['features']['auc']:.4f} "
        f"| {payload['baselines']['features']['brier']:.4f} "
        f"| {payload['baselines']['features']['ece']:.4f} |",
        f"| DistilBERT cross-encoder | "
        f"{payload['baselines']['distilbert']['auc']:.4f} "
        f"| {payload['baselines']['distilbert']['brier']:.4f} "
        f"| {payload['baselines']['distilbert']['ece']:.4f} |",
        f"| **hybrid (this run)** | **{payload['test']['auc']:.4f}** "
        f"| **{payload['test']['brier']:.4f}** "
        f"| **{payload['test']['ece']:.4f}** |",
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(QA_MODEL)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(QA_MODEL).to(device)
    qa_model.eval()

    print("loading contexts...", flush=True)
    train_rows, train_contexts = load_squad_rows(
        "train", args.train_contexts, args.per_context, args.seed)
    cal_rows, cal_contexts = load_squad_rows(
        "train", args.cal_contexts, args.per_context, args.seed + 1,
        exclude_contexts=set(train_contexts))
    test_rows, test_contexts = load_squad_rows(
        "validation", args.test_contexts, args.per_context, args.test_seed,
        max_questions=200)

    with tempfile.TemporaryDirectory(prefix="az-answerability-hybrid-") as folder:
        print(f"extracting {len(train_rows)} train features...", flush=True)
        train = extract_rows(
            train_rows, train_contexts, Path(folder) / "train.db", "train")
        print(f"extracting {len(cal_rows)} calibration features...", flush=True)
        calibration = extract_rows(
            cal_rows, cal_contexts, Path(folder) / "cal.db", "calibration")
        print(f"extracting {len(test_rows)} test features...", flush=True)
        test = extract_rows(
            test_rows, test_contexts, Path(folder) / "test.db", "heldout")
        for split in (train, calibration, test):
            for row in split:
                row["qa"] = qa_features(
                    qa_model, tokenizer, row["question"],
                    row["context_text"], device)

    test_labels = [row["label"] for row in test]
    test_prevalence = sum(test_labels) / len(test_labels)
    print(
        f"fitting hybrid logistic model and prior-weighted isotonic "
        f"calibration (target prevalence {test_prevalence:.3f})...",
        flush=True)
    profile = fit_hybrid(
        train, calibration, target_prevalence=test_prevalence)
    cal_labels = [row["label"] for row in calibration]
    cal_probabilities = predict(calibration, profile)
    cal_raw_scores = raw_probabilities(calibration, profile)
    test_probabilities = predict(test, profile)
    test_raw_scores = raw_probabilities(test, profile)
    selection = select_threshold(cal_labels, cal_probabilities)
    calibration_csv = out / "answerability_calibration.csv"
    write_observations(
        calibration_csv, calibration, cal_probabilities, "calibration",
        cal_raw_scores)
    aisc_calibration = {}
    aisc_threshold = None
    if args.aisc_root and not args.skip_aisc:
        print("running AISC calibration policy...", flush=True)
        aisc_calibration = run_aisc_policy(
            args.aisc_root, calibration_csv,
            out / "aisc_policy_calibration.txt")
        value = (aisc_calibration.get("recommendation") or {}).get("threshold")
        if isinstance(value, (int, float)):
            aisc_threshold = float(value)
    threshold = (args.threshold if args.threshold is not None
                 else aisc_threshold if aisc_threshold is not None
                 else selection["point"]["threshold"])
    test_point = policy_point(test_labels, test_probabilities, threshold)
    test_table = policy_table(test_labels, test_probabilities)
    test_metrics = _metric_summary(test_labels, test_probabilities)
    cal_metrics = _metric_summary(cal_labels, cal_probabilities)
    cal_metrics["split"] = "calibration"
    oof_metrics = out_of_fold_metrics(
        test, test_raw_scores, test_prevalence)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""
    profile.update({
        "n": len(test),
        "brier": test_metrics["brier"],
        "ece": test_metrics["ece"],
        "threshold": threshold,
        "refusal_precision": test_point["refusal_precision"],
        "refusal_recall": test_point["refusal_recall"],
        "answerable_coverage": test_point["answerable_coverage"],
        "wilson": {
            "refusal_precision": test_point["refusal_precision_wilson_95"],
            "refusal_recall": test_point["refusal_recall_wilson_95"],
            "answerable_coverage":
                test_point["answerable_coverage_wilson_95"],
        },
        "metrics": test_metrics,
        "calibration_in_sample": cal_metrics,
        "calibration_oof": oof_metrics,
        # Compatibility key: it now reports the out-of-fold estimate, not the
        # optimistic in-sample fit.
        "calibration_metrics": dict(oof_metrics),
        "selection": {
            "feasible_on_calibration": selection["feasible"],
            "threshold": threshold,
            "misses": selection["misses"],
            "internal_threshold": selection["point"]["threshold"],
            "aisc_threshold": aisc_threshold,
        },
        "n_split": {
            "train": len(train),
            "calibration": len(calibration),
            "heldout": len(test),
        },
        "commit": commit,
        "dataset": "rajpurkar/squad_v2",
        "dataset_hash": hashlib.sha256(
            json.dumps([row["question_id"] for row in test],
                       sort_keys=True).encode("utf-8")).hexdigest(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "calibration_date": time.strftime("%Y-%m-%d"),
        "recalibration_due": time.strftime(
            "%Y-%m-%d", time.localtime(time.time() + 90 * 86400)),
    })
    observations = out / "answerability_observations.csv"
    write_observations(
        observations, test, test_probabilities, "heldout", test_raw_scores)
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
            "stdout": check["stdout"],
            "stderr": check["stderr"],
        }
        profile_path.write_text(
            json.dumps(profile, indent=2) + "\n", encoding="utf-8")

    # Baselines from the earlier runs. They are loaded when present so the
    # report can show the relative strength without pretending they passed.
    baselines = {
        "features": {"auc": 0.5635, "brier": 0.2703, "ece": 0.1407},
        "distilbert": {"auc": 0.5375, "brier": 0.2762, "ece": 0.1612},
    }
    for key, path in (
            ("features", ROOT / "benchmarks/results/answerability_signal/"
                         "answerability_profile.json"),
            ("distilbert", ROOT / "benchmarks/results/answerability_transformer/"
                           "answerability_profile.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            baselines[key] = data.get("metrics", baselines[key])
        except (OSError, json.JSONDecodeError):
            pass

    limitations = [
        "SQuAD v2 no-answer questions share entities and vocabulary with the "
        "source paragraph; this is a stress distribution, not a production "
        "pilot.",
        "The hybrid uses a tiny local QA model as a confidence feature; it is "
        "an experiment artifact, not a default product gate.",
        "The hybrid profile is not loadable by the feature-only optional gate; "
        "that gate uses the profile under answerability_signal.",
        "No threshold meets the 0.90 refusal-recall, 0.90 refusal-precision "
        "and 0.80 answerable-coverage targets together.",
        "CitationGate remains a separate gate and the default refusal "
        "threshold remains 0.5.",
        "Prior weighting corrects the calibration/target prevalence gap but "
        "does not add discriminative signal; the Brier floor is set by the "
        "class overlap, so the 0.15 Brier target is not claimed.",
        "Out-of-fold calibration metrics are a reporting estimate; the "
        "production map is fitted once on the calibration split and is not "
        "refitted on held-out rows.",
        "No 0.98 assurance claim is made.",
    ]
    payload = {
        "commit": commit,
        "dataset": "rajpurkar/squad_v2 validation (50 contexts / 200 questions)",
        "train_n": len(train),
        "calibration_n": len(calibration),
        "calibration": cal_metrics,
        "calibration_method": profile["calibration_method"],
        "prevalence": profile["prevalence"],
        "calibration_in_sample": cal_metrics,
        "calibration_oof": oof_metrics,
        "test": test_metrics,
        "threshold": threshold,
        "test_point": test_point,
        "test_table": test_table,
        "selection": selection,
        "baselines": baselines,
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
        "auc": test_metrics["auc"],
        "calibration_oof_brier": oof_metrics["brier"],
        "calibration_oof_ece": oof_metrics["ece"],
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
