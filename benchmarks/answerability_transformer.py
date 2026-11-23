"""Local DistilBERT cross-encoder baseline for answerability.

This is a stronger-model check, not the shipped signal. It trains a binary
question+context classifier on the same SQuAD v2 train/calibration split used
by ``benchmarks/answerability_signal.py`` and evaluates the same 200-question
held-out sample. The model is local (``distilbert-base-uncased``) and is not
committed; the script retrains it.

Run:

    python benchmarks/answerability_transformer.py \\
        --aisc-root "C:\...\ai-systems-algebra"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from atomzero.doc_memory import DocMemory

try:
    from benchmarks.answerability_signal import (
        TARGETS, THRESHOLDS, _metric_summary, expected_calibration_error,
        extract_rows, load_squad_rows, policy_point, policy_table,
        run_aisc_check, run_aisc_policy, select_threshold,
        write_observations, write_toml)
except ImportError:  # running as `python benchmarks/answerability_transformer.py`
    from answerability_signal import (
        TARGETS, THRESHOLDS, _metric_summary, expected_calibration_error,
        extract_rows, load_squad_rows, policy_point, policy_table,
        run_aisc_check, run_aisc_policy, select_threshold,
        write_observations, write_toml)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = "benchmarks/results/answerability_transformer"
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 384


def _text(row: dict) -> str:
    return f"{row['question']} [SEP] {row.get('context_text', '')}"


def _probabilities(model, tokenizer, rows: list[dict], device,
                   batch_size: int = 32) -> list[float]:
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            encoded = tokenizer(
                [_text(row) for row in batch], truncation=True,
                max_length=MAX_LENGTH, padding="max_length",
                return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits.float(), dim=-1)[:, 1]
            output.extend(float(value) for value in probabilities.cpu())
    return output


def train_transformer(train: list[dict], calibration: list[dict],
                      epochs: int = 2, batch_size: int = 16,
                      learning_rate: float = 2e-5, seed: int = 0):
    try:
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)
    except ImportError as exc:
        raise SystemExit(
            "install the optional neural dependency first: "
            "pip install 'atomzero[neural]'") from exc
    from sklearn.isotonic import IsotonicRegression

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2).to(device)
    encoded = tokenizer(
        [_text(row) for row in train], truncation=True,
        max_length=MAX_LENGTH, padding="max_length", return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    labels = torch.tensor([row["label"] for row in train], dtype=torch.long)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    started = time.time()
    for epoch in range(1, epochs + 1):
        order = torch.randperm(len(train), generator=generator).tolist()
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            batch_ids = input_ids[indices].to(device)
            batch_mask = attention_mask[indices].to(device)
            batch_labels = labels[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=device.type == "cuda"):
                logits = model(input_ids=batch_ids,
                               attention_mask=batch_mask).logits
                loss = F.cross_entropy(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach())
        print(json.dumps({
            "event": "epoch", "epoch": epoch,
            "loss": round(total_loss / max(1, len(order) // batch_size), 4),
            "s": round(time.time() - started, 1)}), flush=True)
    raw_cal = _probabilities(model, tokenizer, calibration, device)
    y_cal = [row["label"] for row in calibration]
    isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    isotonic.fit(raw_cal, y_cal)
    grid = sorted(set([0.0, 1.0] + raw_cal))
    calibrated = isotonic.predict(grid)
    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "calibration": {
            "x": [float(value) for value in grid],
            "y": [float(value) for value in calibrated],
        },
        "raw_cal": raw_cal,
        "train_seconds": round(time.time() - started, 1),
    }


def calibrate(calibration: dict, probabilities: list[float]) -> list[float]:
    x_values = calibration["x"]
    y_values = calibration["y"]
    output = []
    for value in probabilities:
        output.append(max(0.0, min(1.0, float(
            __import__("numpy").interp(value, x_values, y_values)))))
    return output


def write_report(path: Path, payload: dict) -> None:
    lines = [
        "# Local DistilBERT answerability baseline",
        "",
        f"Commit: `{payload['commit']}`",
        f"Dataset / split: {payload['dataset']}",
        "Signal: DistilBERT cross-encoder over question + top retrieved "
        "context; this is a stronger-model baseline, not the shipped gate.",
        "Calibration method: isotonic calibration on a context-grouped "
        "calibration split.",
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
    parser.add_argument("--model-out", default="checkpoints/answerability_distilbert")
    parser.add_argument("--train-contexts", type=int, default=240)
    parser.add_argument("--cal-contexts", type=int, default=60)
    parser.add_argument("--test-contexts", type=int, default=50)
    parser.add_argument("--per-context", type=int, default=5)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--test-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--aisc-root", default="")
    parser.add_argument("--skip-aisc", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args(argv)
    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("loading contexts...", flush=True)
    train_rows, train_contexts = load_squad_rows(
        "train", args.train_contexts, args.per_context, args.seed)
    cal_rows, cal_contexts = load_squad_rows(
        "train", args.cal_contexts, args.per_context, args.seed + 1,
        exclude_contexts=set(train_contexts))
    test_rows, test_contexts = load_squad_rows(
        "validation", args.test_contexts, args.per_context, args.test_seed,
        max_questions=200)

    with tempfile.TemporaryDirectory(prefix="az-answerability-tf-") as folder:
        print(f"extracting {len(train_rows)} train features...", flush=True)
        train = extract_rows(
            train_rows, train_contexts, Path(folder) / "train.db", "train")
        print(f"extracting {len(cal_rows)} calibration features...", flush=True)
        calibration = extract_rows(
            cal_rows, cal_contexts, Path(folder) / "cal.db", "calibration")
        print(f"extracting {len(test_rows)} test features...", flush=True)
        test = extract_rows(
            test_rows, test_contexts, Path(folder) / "test.db", "heldout")

    print("training DistilBERT cross-encoder...", flush=True)
    trained = train_transformer(
        train, calibration, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, seed=args.seed)
    model = trained["model"]
    tokenizer = trained["tokenizer"]
    device = trained["device"]
    raw_cal = trained["raw_cal"]
    y_cal = [row["label"] for row in calibration]
    cal_probabilities = calibrate(trained["calibration"], raw_cal)
    test_raw = _probabilities(model, tokenizer, test, device)
    test_probabilities = calibrate(trained["calibration"], test_raw)
    test_labels = [row["label"] for row in test]
    cal_labels = [row["label"] for row in calibration]

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
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""
    profile = {
        "model": {
            "type": MODEL_NAME,
            "max_length": MAX_LENGTH,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "train_seconds": trained["train_seconds"],
            "weights": str(ROOT / args.model_out),
        },
        "calibration": trained["calibration"],
        "threshold": threshold,
        "metrics": test_metrics,
        "calibration_metrics": cal_metrics,
        "selection": {
            "feasible_on_calibration": selection["feasible"],
            "threshold": threshold,
            "misses": selection["misses"],
            "internal_threshold": selection["point"]["threshold"],
            "aisc_threshold": aisc_threshold,
        },
        "n": {
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
    }
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
            "stdout": check["stdout"],
            "stderr": check["stderr"],
        }
        profile_path.write_text(
            json.dumps(profile, indent=2) + "\n", encoding="utf-8")

    model_path = ROOT / args.model_out
    model_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)

    limitations = [
        "SQuAD v2 no-answer questions share entities and vocabulary with the "
        "source paragraph; this is a stress distribution.",
        "The DistilBERT cross-encoder is a local baseline; it is not shipped "
        "in the default product package and is not enabled by default.",
        "The target refusal metrics are still reported as pass/fail, not as "
        "an assurance claim.",
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
        "model": str(model_path),
        "train_n": len(train),
        "calibration_n": len(calibration),
        "test_n": len(test),
        "threshold": threshold,
        "brier": test_metrics["brier"],
        "ece": test_metrics["ece"],
        "auc": test_metrics["auc"],
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
