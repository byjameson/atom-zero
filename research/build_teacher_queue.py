"""Build a small verified teacher queue with a local Qwen teacher.

The teacher is a real local model (Qwen2.5-1.5B-Instruct by default). Answers are
kept only when a deterministic format validator accepts them. The queue has
three distributions so a later adapter run can separate in-distribution from
cross-distribution held-out data:

- ``json``: compact JSON with name/score keys
- ``keyvalue``: one-line ``name=...;score=...`` form

The current local 1.5B teacher reliably follows the JSON and key-value
formats. CSV is supported by the validator but disabled by default because the
small teacher tended to invent a different row instead of copying the requested
values.

This is a research fixture generator; it does not train or apply an adapter.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NAMES = [
    "Ada", "Bora", "Cem", "Derya", "Ege", "Funda", "Gizem", "Hakan",
    "Ipek", "Jale", "Kaan", "Lale", "Mert", "Naz", "Ozan", "Pelin",
    "Rana", "Selin", "Tolga", "Umut", "Vera", "Yasin", "Zeynep", "Aylin",
]
DISTRIBUTIONS = ("json", "csv", "keyvalue")


def _prompt(distribution: str, name: str, score: int) -> str:
    if distribution == "json":
        return (f'Return only compact JSON with keys "name" and "score" for '
                f"name={name} and score={score}.")
    if distribution == "csv":
        return ("Output exactly two lines and nothing else, with no prose "
                "and no code fence:\n"
                f"name,score\n{name},{score}")
    return ("Output exactly one line and nothing else, with no prose and no "
            f"code fence: name={name};score={score}")


def _clean(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _valid(distribution: str, answer: str, name: str, score: int) -> bool:
    answer = _clean(answer)
    if distribution == "json":
        try:
            data = json.loads(answer)
        except json.JSONDecodeError:
            return False
        return (isinstance(data, dict)
                and str(data.get("name")) == name
                and int(data.get("score")) == score)
    if distribution == "csv":
        lines = [line.strip() for line in answer.splitlines() if line.strip()]
        if len(lines) < 2:
            return False
        header = [part.strip().lower() for part in lines[0].split(",")]
        row = [part.strip() for part in lines[1].split(",")]
        return (header[:2] == ["name", "score"] and len(row) >= 2
                and row[0] == name and row[1].isdigit()
                and int(row[1]) == score)
    match = re.search(
        r"name\s*=\s*([^;,\n]+)\s*[;,\s]\s*score\s*=\s*(\d+)", answer,
        re.IGNORECASE)
    return bool(match and match.group(1).strip() == name
                and int(match.group(2)) == score)


def _generate(model, tok, prompt: str, max_new_tokens: int, device) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.eos_token_id)
    generated = output[0][inputs["input_ids"].shape[1]:]
    return _clean(tok.decode(generated, skip_special_tokens=True))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=r"C:\Users\biris\Desktop\opencode\ai gelistir\models"
                r"\Qwen2.5-1.5B-Instruct")
    parser.add_argument("--out", default="data/teacher_queue.jsonl")
    parser.add_argument("--rejected", default="data/teacher_queue_rejected.jsonl")
    parser.add_argument("--json-count", type=int, default=64)
    parser.add_argument("--csv-count", type=int, default=0)
    parser.add_argument("--keyvalue-count", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=str(device))
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    rng = random.Random(args.seed)
    counts = {
        "json": args.json_count,
        "csv": args.csv_count,
        "keyvalue": args.keyvalue_count,
    }
    accepted, rejected = [], []
    index = 0
    for distribution in DISTRIBUTIONS:
        for _ in range(counts[distribution]):
            name = NAMES[index % len(NAMES)]
            score = rng.randint(1, 99)
            prompt = _prompt(distribution, name, score)
            answer = _generate(
                model, tok, prompt, args.max_new_tokens, device)
            row = {
                "user": prompt, "assistant": answer,
                "distribution": distribution,
                "verification": {
                    "verified": _valid(distribution, answer, name, score),
                    "promotable": False,
                    "status": "verified",
                    "method": "deterministic_teacher_format_validator_v1",
                },
            }
            row["verification"]["promotable"] = row["verification"]["verified"]
            if row["verification"]["verified"]:
                accepted.append(row)
            else:
                rejected.append(row)
            index += 1
            if index % 10 == 0:
                print(json.dumps({
                    "generated": index,
                    "accepted": len(accepted),
                    "rejected": len(rejected),
                    "device": str(device),
                }), flush=True)
    for path, rows in ((args.out, accepted), (args.rejected, rejected)):
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n"
                    for row in rows), encoding="utf-8")
    print(json.dumps({
        "accepted": len(accepted),
        "rejected": len(rejected),
        "accepted_by_distribution": {
            dist: sum(1 for row in accepted
                      if row["distribution"] == dist)
            for dist in DISTRIBUTIONS
        },
        "out": str(args.out),
        "device": str(device),
        "teacher": args.model,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
