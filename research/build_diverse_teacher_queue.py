"""Build a diverse multi-family teacher queue with a local Qwen teacher.

Training families:
- person_json / product_json (JSON format, two key families)
- person_kv / product_kv (key-value format, two key families)

Held-out family:
- city_json / city_kv (new entity/key family, both formats)

The training mix keeps the original JSON family as replay while adding a second
format and a second key family. The held-out family is never mixed into
training.

    python research/build_diverse_teacher_queue.py
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PEOPLE = [
    "Ada", "Bora", "Cem", "Derya", "Ege", "Funda", "Gizem", "Hakan",
    "Ipek", "Jale", "Kaan", "Lale", "Mert", "Naz", "Ozan", "Pelin",
    "Rana", "Selin", "Tolga", "Umut",
]
PRODUCTS = [
    "notebook", "lamp", "chair", "phone", "keyboard", "monitor",
    "backpack", "bottle", "camera", "speaker", "tablet", "watch",
]
CITIES = [
    "Ankara", "Istanbul", "Izmir", "Bursa", "Antalya", "Konya",
    "Adana", "Gaziantep", "Trabzon", "Kayseri", "Mersin", "Eskisehir",
]
FAMILIES = {
    "person_json": (PEOPLE, "name", "score", "json"),
    "product_json": (PRODUCTS, "product", "price", "json"),
    "person_kv": (PEOPLE, "name", "score", "kv"),
    "product_kv": (PRODUCTS, "product", "price", "kv"),
    "city_json": (CITIES, "city", "population", "json"),
    "city_kv": (CITIES, "city", "population", "kv"),
}
TRAIN_FAMILIES = ("person_json", "product_json", "person_kv", "product_kv")
CROSS_FAMILIES = ("city_json", "city_kv")


def _prompt(family: str, entity: str, number: int) -> str:
    _, key1, key2, kind = FAMILIES[family]
    if kind == "json":
        return (f'Return only compact JSON with keys "{key1}" and "{key2}" '
                f"for {key1}={entity} and {key2}={number}.")
    return ("Output exactly one line and nothing else, with no prose and no "
            f"code fence: {key1}={entity};{key2}={number}")


def _clean(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _valid(family: str, answer: str, entity: str, number: int) -> bool:
    _, key1, key2, kind = FAMILIES[family]
    answer = _clean(answer)
    if kind == "json":
        try:
            data = json.loads(answer)
        except json.JSONDecodeError:
            return False
        return (isinstance(data, dict)
                and str(data.get(key1)) == entity
                and int(data.get(key2)) == number)
    pattern = (rf"{re.escape(key1)}\s*=\s*([^;,\n]+)\s*[;,\s]\s*"
               rf"{re.escape(key2)}\s*=\s*(\d+)")
    match = re.search(pattern, answer, re.IGNORECASE)
    return bool(match and match.group(1).strip() == entity
                and int(match.group(2)) == number)


def _generate(model, tok, prompt: str, max_new_tokens: int, device) -> str:
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.eos_token_id)
    return _clean(tok.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=r"C:\Users\biris\Desktop\opencode\ai gelistir\models"
                r"\Qwen2.5-1.5B-Instruct")
    parser.add_argument("--out-dir", default="data/diverse_teacher")
    parser.add_argument("--per-family", type=int, default=40)
    parser.add_argument("--city-per-family", type=int, default=25)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=str(device))
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    rng = random.Random(args.seed)
    rows = []
    accepted = 0
    total = 0
    for family in TRAIN_FAMILIES + CROSS_FAMILIES:
        entities, key1, key2, kind = FAMILIES[family]
        count = (args.city_per_family if family in CROSS_FAMILIES
                 else args.per_family)
        for index in range(count):
            entity = entities[index % len(entities)]
            number = rng.randint(20, 900) if key2 == "population" else (
                rng.randint(1, 99) if key2 == "score" else rng.randint(5, 500))
            prompt = _prompt(family, entity, number)
            answer = _generate(
                model, tok, prompt, args.max_new_tokens, device)
            verified = _valid(family, answer, entity, number)
            rows.append({
                "user": prompt,
                "assistant": answer,
                "family": family,
                "verification": {
                    "verified": verified,
                    "promotable": verified,
                    "status": "verified" if verified else "unverified",
                    "method": "deterministic_diverse_format_validator_v1",
                },
            })
            total += 1
            accepted += int(verified)
            if total % 20 == 0:
                print(json.dumps({
                    "generated": total, "accepted": accepted,
                    "rejected": total - accepted,
                    "family": family, "device": str(device),
                }), flush=True)
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    verified = [row for row in rows if row["verification"]["verified"]]
    train = []
    indist = []
    for family in TRAIN_FAMILIES:
        subset = [row for row in verified if row["family"] == family]
        cut = max(1, int(round(len(subset) * 0.8)))
        train.extend(subset[:cut])
        indist.extend(subset[cut:])
    cross = [row for row in verified if row["family"] in CROSS_FAMILIES]
    files = {
        "train": train, "indist": indist, "cross": cross,
        "all": verified,
    }
    for name, subset in files.items():
        target = out_dir / f"{name}.jsonl"
        target.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n"
                    for row in subset), encoding="utf-8")
    print(json.dumps({
        "requested": total,
        "accepted": len(verified),
        "rejected": total - len(verified),
        "train": len(train),
        "indist": len(indist),
        "cross": len(cross),
        "by_family": {
            family: sum(1 for row in verified
                        if row["family"] == family)
            for family in FAMILIES
        },
        "out_dir": str(out_dir),
        "device": str(device),
        "teacher": args.model,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
