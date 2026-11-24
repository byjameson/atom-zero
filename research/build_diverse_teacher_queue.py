"""Build a diverse multi-family teacher queue with a local Qwen teacher.

Training families:
- person_json / product_json (JSON, name/score and product/price)
- person_kv / product_kv (key-value, two key families)
- arith_plain / arith_json (small addition, plain sentence and JSON)

Held-out families:
- city_json / city_kv (new entity/key family, JSON and key-value)
- city_yaml (new entity/key family and a YAML-like format)

Family definitions and validators are importable without torch/transformers.
The model is loaded lazily inside ``main`` (or not at all with ``--dry-run``),
so tests can drive ``build_rows`` with a stub teacher:

    rows = build_rows(["city_yaml"], stub_teacher, count=2, seed=17)

The training mix keeps the original JSON family as replay while adding a
second format, a second key family and arithmetic. Held-out families are never
mixed into training.

    python research/build_diverse_teacher_queue.py
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

PEOPLE = (
    "Ada", "Bora", "Cem", "Derya", "Ege", "Funda", "Gizem", "Hakan",
    "Ipek", "Jale", "Kaan", "Lale", "Mert", "Naz", "Ozan", "Pelin",
    "Rana", "Selin", "Tolga", "Umut",
)
PRODUCTS = (
    "notebook", "lamp", "chair", "phone", "keyboard", "monitor",
    "backpack", "bottle", "camera", "speaker", "tablet", "watch",
)
CITIES = (
    "Ankara", "Istanbul", "Izmir", "Bursa", "Antalya", "Konya",
    "Adana", "Gaziantep", "Trabzon", "Kayseri", "Mersin", "Eskisehir",
)

VERIFICATION_METHOD = "deterministic_diverse_format_validator_v1"
FORMATS = ("json", "kv", "yaml", "plain")
ARITH_FAMILIES = ("arith_plain", "arith_json")

PromptBuilder = Callable[["FamilySpec", str, int], str]


@dataclass(frozen=True)
class FamilySpec:
    """One teacher task family.

    ``entity`` is the string value for ``entity_name`` and ``number`` is the
    numeric value for ``key2``. Arithmetic families use ``entity`` as the left
    operand and ``number`` as the right operand; the expected answer is the
    sum. ``key2`` is ``None`` when the format has a single output key.
    """

    name: str
    entity_name: str
    key1: str
    key2: str | None
    format: str
    entities: tuple[str, ...]
    number_range: tuple[int, int]
    prompt_builder: PromptBuilder
    arithmetic: bool = False


def _prompt_json(spec: FamilySpec, entity: str, number: int) -> str:
    return (f'Return only compact JSON with keys "{spec.key1}" and '
            f'"{spec.key2}" for {spec.key1}={entity} and '
            f"{spec.key2}={number}.")


def _prompt_kv(spec: FamilySpec, entity: str, number: int) -> str:
    return ("Output exactly one line and nothing else, with no prose and no "
            f"code fence: {spec.key1}={entity};{spec.key2}={number}")


def _prompt_yaml(spec: FamilySpec, entity: str, number: int) -> str:
    return ("Output exactly two YAML-like lines and nothing else, with no "
            "prose, no braces and no code fence:\n"
            f"{spec.key1}: {entity}\n{spec.key2}: {number}")


def _prompt_arith_plain(spec: FamilySpec, entity: str, number: int) -> str:
    return (f"What is {entity} + {number}? Output exactly one line and "
            "nothing else, with no prose and no code fence. Format example: "
            "The sum of 2 and 3 is 5.")


def _prompt_arith_json(spec: FamilySpec, entity: str, number: int) -> str:
    return (f'What is {entity} + {number}? Return only compact JSON with '
            f'key "{spec.key1}", for example {{"{spec.key1}": 5}}.')


FAMILIES: dict[str, FamilySpec] = {
    "person_json": FamilySpec(
        name="person_json", entity_name="name", key1="name", key2="score",
        format="json", entities=PEOPLE, number_range=(1, 99),
        prompt_builder=_prompt_json),
    "product_json": FamilySpec(
        name="product_json", entity_name="product", key1="product",
        key2="price", format="json", entities=PRODUCTS,
        number_range=(5, 500), prompt_builder=_prompt_json),
    "person_kv": FamilySpec(
        name="person_kv", entity_name="name", key1="name", key2="score",
        format="kv", entities=PEOPLE, number_range=(1, 99),
        prompt_builder=_prompt_kv),
    "product_kv": FamilySpec(
        name="product_kv", entity_name="product", key1="product",
        key2="price", format="kv", entities=PRODUCTS,
        number_range=(5, 500), prompt_builder=_prompt_kv),
    "arith_plain": FamilySpec(
        name="arith_plain", entity_name="operand", key1="sum", key2=None,
        format="plain", entities=(), number_range=(1, 99),
        prompt_builder=_prompt_arith_plain, arithmetic=True),
    "arith_json": FamilySpec(
        name="arith_json", entity_name="operand", key1="sum", key2=None,
        format="json", entities=(), number_range=(1, 99),
        prompt_builder=_prompt_arith_json, arithmetic=True),
    "city_json": FamilySpec(
        name="city_json", entity_name="city", key1="city", key2="population",
        format="json", entities=CITIES, number_range=(50_000, 2_000_000),
        prompt_builder=_prompt_json),
    "city_kv": FamilySpec(
        name="city_kv", entity_name="city", key1="city", key2="population",
        format="kv", entities=CITIES, number_range=(50_000, 2_000_000),
        prompt_builder=_prompt_kv),
    "city_yaml": FamilySpec(
        name="city_yaml", entity_name="city", key1="city", key2="population",
        format="yaml", entities=CITIES, number_range=(50_000, 2_000_000),
        prompt_builder=_prompt_yaml),
}
TRAIN_FAMILIES = (
    "person_json", "product_json", "person_kv", "product_kv",
    "arith_plain", "arith_json",
)
CROSS_FAMILIES = ("city_json", "city_kv", "city_yaml")
DEFAULT_COUNTS = {
    "person_json": 40, "product_json": 40, "person_kv": 40,
    "product_kv": 40, "city_json": 25, "city_kv": 25, "city_yaml": 25,
    "arith_plain": 20, "arith_json": 20,
}


def get_family(family: str | FamilySpec) -> FamilySpec:
    if isinstance(family, FamilySpec):
        return family
    try:
        return FAMILIES[family]
    except KeyError:
        raise KeyError(f"unknown teacher family: {family}") from None


def _clean(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def clean_answer(text: str) -> str:
    """Strip surrounding whitespace and an optional Markdown code fence."""

    return _clean(text)


def _as_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _expected(spec: FamilySpec, entity: str, number: int) -> tuple[str, int]:
    if spec.arithmetic:
        total = int(entity) + int(number)
        return str(total), total
    return entity, int(number)


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _validate_json(spec: FamilySpec, answer: str, entity: str,
                   number: int) -> bool:
    try:
        data = json.loads(answer)
    except json.JSONDecodeError:
        return False
    if not isinstance(data, dict):
        return False
    expected_text, expected_number = _expected(spec, entity, number)
    if spec.key1 not in data or str(data.get(spec.key1)) != expected_text:
        return False
    if spec.key2 is None:
        return True
    return _as_int(data.get(spec.key2)) == expected_number


def _validate_kv(spec: FamilySpec, answer: str, entity: str,
                 number: int) -> bool:
    if spec.key2 is None:
        raise ValueError(f"kv family {spec.name} needs key2")
    pattern = (rf"{re.escape(spec.key1)}\s*=\s*([^;,\n]+)\s*[;,\s]\s*"
               rf"{re.escape(spec.key2)}\s*=\s*(-?\d+)")
    match = re.search(pattern, answer, re.IGNORECASE)
    if not match:
        return False
    expected_text, expected_number = _expected(spec, entity, number)
    return (match.group(1).strip() == expected_text
            and int(match.group(2)) == expected_number)


def _validate_yaml(spec: FamilySpec, answer: str, entity: str,
                   number: int) -> bool:
    if spec.key2 is None:
        raise ValueError(f"yaml family {spec.name} needs key2")
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if len(lines) != 2:
        return False
    values: dict[str, str] = {}
    for line in lines:
        if "{" in line or "}" in line or "[" in line or line.startswith("-"):
            return False
        key, separator, value = line.partition(":")
        if not separator:
            return False
        key = key.strip()
        value = value.strip()
        if not key or not value or key in values:
            return False
        values[key] = value
    if set(values) != {spec.key1, spec.key2}:
        return False
    expected_text, expected_number = _expected(spec, entity, number)
    return (_unquote(values[spec.key1]) == expected_text
            and _as_int(_unquote(values[spec.key2])) == expected_number)


def _validate_plain(spec: FamilySpec, answer: str, entity: str,
                    number: int) -> bool:
    if not spec.arithmetic:
        raise ValueError(f"plain family {spec.name} must be arithmetic")
    expected_number = int(entity) + int(number)
    pattern = (rf"The sum of {re.escape(str(entity))} and "
               rf"{re.escape(str(number))} is {expected_number}\s*\.?")
    return bool(re.fullmatch(pattern, answer, re.IGNORECASE))


FORMAT_VALIDATORS: dict[
    str, Callable[[FamilySpec, str, str, int], bool]] = {
    "json": _validate_json,
    "kv": _validate_kv,
    "yaml": _validate_yaml,
    "plain": _validate_plain,
}


def validate_answer(family: str | FamilySpec, answer: str, entity: str,
                    number: int) -> bool:
    """Return True when ``answer`` is the requested format and values.

    This checks format and requested values, not factual truth. It is
    deterministic and uses only the standard library.
    """

    spec = get_family(family)
    validator = FORMAT_VALIDATORS.get(spec.format)
    if validator is None:
        raise ValueError(f"unknown format for {spec.name}: {spec.format}")
    answer = _clean(answer)
    if not answer:
        return False
    return validator(spec, answer, entity, number)


def oracle_answer(family: str | FamilySpec, entity: str, number: int) -> str:
    """Format the expected answer without a teacher model."""

    spec = get_family(family)
    expected_text, expected_number = _expected(spec, entity, number)
    if spec.format == "json":
        payload = {spec.key1: expected_number if spec.arithmetic
                   else expected_text}
        if spec.key2 is not None:
            payload[spec.key2] = expected_number
        return json.dumps(payload)
    if spec.format == "kv":
        return f"{spec.key1}={expected_text};{spec.key2}={expected_number}"
    if spec.format == "yaml":
        return f"{spec.key1}: {expected_text}\n{spec.key2}: {expected_number}"
    if spec.format == "plain":
        return (f"The sum of {entity} and {number} is {expected_number}.")
    raise ValueError(f"unknown format for {spec.name}: {spec.format}")


def _case(spec: FamilySpec, index: int, rng: random.Random) -> tuple[str, int]:
    if spec.arithmetic:
        left = rng.randint(*spec.number_range)
        right = rng.randint(*spec.number_range)
        return str(left), right
    if not spec.entities:
        raise ValueError(f"family {spec.name} has no entities")
    entity = spec.entities[index % len(spec.entities)]
    return entity, rng.randint(*spec.number_range)


def _resolve_row_counts(families: Sequence[str],
                        counts: Mapping[str, int] | None,
                        count: int | None) -> dict[str, int]:
    overrides = dict(counts or {})
    resolved: dict[str, int] = {}
    for family in families:
        if family not in FAMILIES:
            raise KeyError(f"unknown teacher family: {family}")
        value = overrides.get(
            family, count if count is not None
            else DEFAULT_COUNTS.get(family, 20))
        if int(value) < 0:
            raise ValueError(f"count for {family} must be non-negative")
        resolved[family] = int(value)
    return resolved


def build_rows(families: Iterable[str],
               teacher_call: Callable[[str], str] | None = None, *,
               count: int | None = None,
               counts: Mapping[str, int] | None = None,
               seed: int = 17,
               progress: Callable[[dict], None] | None = None) -> list[dict]:
    """Generate and validate rows for ``families``.

    ``teacher_call`` receives the prompt and returns raw text. Pass ``None`` to
    use the deterministic oracle formatter, which is useful for tests and
    ``--dry-run``. No model is loaded here.
    """

    family_list = [families] if isinstance(families, str) else list(families)
    row_counts = _resolve_row_counts(family_list, counts, count)
    rng = random.Random(seed)
    rows: list[dict] = []
    total = 0
    accepted = 0
    for family in family_list:
        spec = FAMILIES[family]
        for index in range(row_counts[family]):
            entity, number = _case(spec, index, rng)
            prompt = spec.prompt_builder(spec, entity, number)
            answer = (oracle_answer(spec, entity, number)
                      if teacher_call is None else teacher_call(prompt))
            answer = _clean(answer)
            verified = validate_answer(spec, answer, entity, number)
            rows.append({
                "user": prompt,
                "assistant": answer,
                "family": family,
                "verification": {
                    "verified": verified,
                    "promotable": verified,
                    "status": "verified" if verified else "unverified",
                    "method": VERIFICATION_METHOD,
                },
            })
            total += 1
            accepted += int(verified)
            if progress is not None and total % 20 == 0:
                progress({
                    "generated": total, "accepted": accepted,
                    "rejected": total - accepted, "family": family,
                })
    return rows


def verified_rows(rows: Iterable[dict]) -> list[dict]:
    """Return only rows whose deterministic validator accepted the answer."""

    return [row for row in rows if row["verification"]["verified"]]


def split_rows(rows: Sequence[dict], *,
               train_families: Sequence[str] = TRAIN_FAMILIES,
               cross_families: Sequence[str] = CROSS_FAMILIES,
               train_ratio: float = 0.8) -> dict[str, list[dict]]:
    """Split verified rows into train/indist/cross/all without family leaks."""

    overlap = set(train_families) & set(cross_families)
    if overlap:
        raise ValueError(
            f"train and cross families overlap: {sorted(overlap)}")
    if not 0.0 <= train_ratio <= 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    verified = verified_rows(rows)
    train: list[dict] = []
    indist: list[dict] = []
    for family in train_families:
        subset = [row for row in verified if row["family"] == family]
        if not subset:
            continue
        cut = max(1, int(round(len(subset) * train_ratio)))
        train.extend(subset[:cut])
        indist.extend(subset[cut:])
    cross_set = set(cross_families)
    cross = [row for row in verified if row["family"] in cross_set]
    return {"train": train, "indist": indist, "cross": cross, "all": verified}


def write_rows(files: Mapping[str, Sequence[dict]],
               out_dir: str | Path) -> dict[str, Path]:
    target_dir = Path(out_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, subset in files.items():
        target = target_dir / f"{name}.jsonl"
        target.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n"
                    for row in subset), encoding="utf-8")
        written[name] = target
    return written


def resolve_counts(families: Iterable[str], *,
                   per_family: int = 40,
                   city_per_family: int = 25,
                   city_yaml_count: int | None = None,
                   arith_count: int = 20,
                   overrides: Mapping[str, int] | None = None
                   ) -> dict[str, int]:
    """Resolve CLI counts to one count per family."""

    family_list = [families] if isinstance(families, str) else list(families)
    if isinstance(overrides, Mapping):
        for family in overrides:
            if family not in FAMILIES:
                raise KeyError(f"unknown teacher family: {family}")
    counts: dict[str, int] = {}
    for family in family_list:
        if family not in FAMILIES:
            raise KeyError(f"unknown teacher family: {family}")
        if overrides and family in overrides:
            value = overrides[family]
        elif family in ARITH_FAMILIES:
            value = arith_count
        elif family == "city_yaml":
            value = (city_per_family if city_yaml_count is None
                     else city_yaml_count)
        elif family in CROSS_FAMILIES:
            value = city_per_family
        else:
            value = per_family
        if int(value) < 0:
            raise ValueError(f"count for {family} must be non-negative")
        counts[family] = int(value)
    return counts


def _generate(model, tok, prompt: str, max_new_tokens: int, device) -> str:
    import torch

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


def _make_teacher(model_path: str, max_new_tokens: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=str(device))
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    def teacher_call(prompt: str) -> str:
        return _generate(model, tok, prompt, max_new_tokens, device)

    return teacher_call, str(device)


def _family_count_arg(value: str) -> tuple[str, int]:
    name, separator, raw = value.partition("=")
    name = name.strip()
    if not separator or not name:
        raise argparse.ArgumentTypeError("expected FAMILY=COUNT")
    try:
        count = int(raw.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"count must be an integer: {value}") from None
    if count < 0:
        raise argparse.ArgumentTypeError("count must be non-negative")
    return name, count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=r"C:\Users\biris\Desktop\opencode\ai gelistir\models"
                r"\Qwen2.5-1.5B-Instruct")
    parser.add_argument("--out-dir", default="data/diverse_teacher")
    parser.add_argument("--per-family", type=int, default=40)
    parser.add_argument("--city-per-family", type=int, default=25)
    parser.add_argument(
        "--city-yaml-count", type=int, default=None,
        help="city_yaml rows; defaults to --city-per-family")
    parser.add_argument(
        "--arith-count", type=int, default=20,
        help="rows per arithmetic family")
    parser.add_argument(
        "--family-count", action="append", default=[], type=_family_count_arg,
        metavar="FAMILY=COUNT", help="override one family count; repeatable")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="use the deterministic oracle formatter, no model load")
    args = parser.parse_args(argv)

    overrides = dict(args.family_count)
    unknown = sorted(set(overrides) - set(FAMILIES))
    if unknown:
        parser.error(f"unknown family in --family-count: {', '.join(unknown)}")
    counts = resolve_counts(
        TRAIN_FAMILIES + CROSS_FAMILIES,
        per_family=args.per_family,
        city_per_family=args.city_per_family,
        city_yaml_count=args.city_yaml_count,
        arith_count=args.arith_count,
        overrides=overrides,
    )

    if args.dry_run:
        teacher_call = None
        device = "none"
        teacher_name = "oracle(dry-run)"
    else:
        teacher_call, device = _make_teacher(
            args.model, args.max_new_tokens)
        teacher_name = args.model

    def progress(event: dict) -> None:
        print(json.dumps(dict(event, device=device)), flush=True)

    rows = build_rows(
        TRAIN_FAMILIES + CROSS_FAMILIES, teacher_call,
        counts=counts, seed=args.seed, progress=progress)
    files = split_rows(rows)
    written = write_rows(files, args.out_dir)
    verified = files["all"]
    print(json.dumps({
        "requested": len(rows),
        "accepted": len(verified),
        "rejected": len(rows) - len(verified),
        "train": len(files["train"]),
        "indist": len(files["indist"]),
        "cross": len(files["cross"]),
        "by_family": {
            family: sum(1 for row in verified if row["family"] == family)
            for family in FAMILIES
        },
        "counts": counts,
        "out_dir": str(Path(args.out_dir).expanduser()),
        "files": {name: str(path) for name, path in written.items()},
        "device": device,
        "teacher": teacher_name,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
