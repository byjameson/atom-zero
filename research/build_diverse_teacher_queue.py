"""Build a diverse multi-family teacher queue with a local Qwen teacher.

Training families:
- person_json / product_json / person_yaml / product_yaml (JSON and YAML)
- person_kv / product_kv (key-value)
- arith_plain / arith_json (small addition, plain sentence and JSON)

Held-out families:
- city_json / city_kv / city_yaml (new entity/key family, all three formats)

Copy-generalization curriculum:
- ``KEY_PAIRS`` lists twelve key pairs. ``--key-pairs`` generates extra
  JSON/KV families (for example ``label_json`` and ``label_kv`` with
  ``label``/``points``), so the model must copy keys from the prompt instead
  of memorizing only ``name``/``score`` and ``product``/``price``.
- ``--holdout-keys`` makes every family containing a key eval-only.
- ``--canonical-json`` re-emits verified JSON answers with compact separators
  and KV answers without spaces, matching what the prompt asks for.
- ``--split-by entity`` assigns entities to train/eval before emitting rows;
  JSON and KV families that share an entity pool always get the same
  assignment. The build summary reports ``entities_train``,
  ``entities_eval`` and ``entity_overlap`` (0 for an entity split).
- ``--replay-out PATH`` writes legacy-family training rows to a separate
  replay file and removes them from ``train.jsonl``, so a replay mix never
  duplicates a primary train row.
- ``--shuffle`` deterministically shuffles generated rows with ``--seed``.

Family definitions and validators are importable without torch/transformers.
The model is loaded lazily inside ``main`` (or not at all with ``--dry-run``),
so tests can drive ``build_rows`` with a stub teacher:

    rows = build_rows(["city_yaml"], stub_teacher, count=2, seed=17)

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
LABELS = (
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta",
    "eta", "theta", "iota", "kappa", "lambda", "omega",
)
CODES = (
    "A17", "B42", "C08", "D31", "E55", "F09",
    "G64", "H23", "J88", "K12", "L47", "M06",
)
TEAMS = (
    "Falcons", "Tigers", "Wolves", "Comets", "Ravens", "Sharks",
    "Dragons", "Hawks", "Bears", "Lions", "Pumas", "Eagles",
)
AUTHORS = (
    "Avery", "Blake", "Carter", "Dawson", "Ellis", "Foster",
    "Griffin", "Hayes", "Ingram", "Jensen", "Keller", "Lawson",
)
SONGS = (
    "Blue River", "Night Drive", "Paper Moon", "Golden Hour",
    "Silent Rain", "City Lights", "Slow Fire", "Ocean Glass",
    "Red Sky", "Winter Sun", "Open Road", "First Light",
)
MODELS = (
    "Atlas", "Beacon", "Cobalt", "Delta", "Ember", "Falcon",
    "Gemini", "Helios", "Ion", "Juno", "Kestrel", "Lumen",
)
RIVERS = (
    "Danube", "Rhine", "Amazon", "Nile", "Volga", "Ganges",
    "Mekong", "Congo", "Niger", "Orinoco", "Yangtze", "Murray",
)
PLANETS = (
    "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn",
    "Uranus", "Neptune", "Ceres", "Pluto", "Eris", "Titan",
)
RECIPES = (
    "lentil soup", "flatbread", "rice pilaf", "tomato stew",
    "oat cookies", "bean salad", "mushroom pasta", "apple cake",
    "yogurt dip", "grilled fish", "vegetable curry", "cheese omelet",
)

KEY_PAIRS = (
    ("name", "score"),
    ("product", "price"),
    ("label", "points"),
    ("code", "rank"),
    ("team", "wins"),
    ("author", "year"),
    ("song", "length"),
    ("model", "params"),
    ("river", "km"),
    ("planet", "mass"),
    ("recipe", "minutes"),
    ("city", "population"),
)
ENTITY_POOLS = {
    "person": PEOPLE,
    "product": PRODUCTS,
    "city": CITIES,
    "label": LABELS,
    "code": CODES,
    "team": TEAMS,
    "author": AUTHORS,
    "song": SONGS,
    "model": MODELS,
    "river": RIVERS,
    "planet": PLANETS,
    "recipe": RECIPES,
}
PAIR_POOLS = {
    "name": "person",
    "product": "product",
    "city": "city",
}
PAIR_NUMBER_RANGES = {
    ("name", "score"): (1, 99),
    ("product", "price"): (5, 500),
    ("label", "points"): (1, 100),
    ("code", "rank"): (1, 50),
    ("team", "wins"): (0, 30),
    ("author", "year"): (1900, 2025),
    ("song", "length"): (60, 360),
    ("model", "params"): (1, 400),
    ("river", "km"): (100, 7000),
    ("planet", "mass"): (1, 1000),
    ("recipe", "minutes"): (5, 180),
    ("city", "population"): (50_000, 2_000_000),
}

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
    ``pool`` names the shared entity pool so JSON and KV families can be split
    consistently at the entity level.
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
    pool: str | None = None


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
        prompt_builder=_prompt_json, pool="person"),
    "product_json": FamilySpec(
        name="product_json", entity_name="product", key1="product",
        key2="price", format="json", entities=PRODUCTS,
        number_range=(5, 500), prompt_builder=_prompt_json, pool="product"),
    "person_kv": FamilySpec(
        name="person_kv", entity_name="name", key1="name", key2="score",
        format="kv", entities=PEOPLE, number_range=(1, 99),
        prompt_builder=_prompt_kv, pool="person"),
    "product_kv": FamilySpec(
        name="product_kv", entity_name="product", key1="product",
        key2="price", format="kv", entities=PRODUCTS,
        number_range=(5, 500), prompt_builder=_prompt_kv, pool="product"),
    "person_yaml": FamilySpec(
        name="person_yaml", entity_name="name", key1="name", key2="score",
        format="yaml", entities=PEOPLE, number_range=(1, 99),
        prompt_builder=_prompt_yaml, pool="person"),
    "product_yaml": FamilySpec(
        name="product_yaml", entity_name="product", key1="product",
        key2="price", format="yaml", entities=PRODUCTS,
        number_range=(5, 500), prompt_builder=_prompt_yaml, pool="product"),
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
        prompt_builder=_prompt_json, pool="city"),
    "city_kv": FamilySpec(
        name="city_kv", entity_name="city", key1="city", key2="population",
        format="kv", entities=CITIES, number_range=(50_000, 2_000_000),
        prompt_builder=_prompt_kv, pool="city"),
    "city_yaml": FamilySpec(
        name="city_yaml", entity_name="city", key1="city", key2="population",
        format="yaml", entities=CITIES, number_range=(50_000, 2_000_000),
        prompt_builder=_prompt_yaml, pool="city"),
}
BASE_PAIR_FAMILIES = {
    ("name", "score"): ("person_json", "person_kv", "person_yaml"),
    ("product", "price"): ("product_json", "product_kv", "product_yaml"),
    ("city", "population"): ("city_json", "city_kv", "city_yaml"),
}


def _register_extra_pair_families() -> None:
    """Add JSON/KV families for key pairs that lack a hand-written family."""

    for key1, key2 in KEY_PAIRS:
        if (key1, key2) in BASE_PAIR_FAMILIES:
            continue
        pool = PAIR_POOLS.get(key1, key1)
        entities = ENTITY_POOLS[pool]
        number_range = PAIR_NUMBER_RANGES[(key1, key2)]
        for fmt, builder in (("json", _prompt_json), ("kv", _prompt_kv)):
            name = f"{key1}_{fmt}"
            if name in FAMILIES:
                raise RuntimeError(f"duplicate generated family: {name}")
            FAMILIES[name] = FamilySpec(
                name=name, entity_name=key1, key1=key1, key2=key2,
                format=fmt, entities=entities, number_range=number_range,
                prompt_builder=builder, pool=pool)


_register_extra_pair_families()

PAIR_FAMILIES: dict[tuple[str, str], tuple[str, ...]] = {}
for _pair in KEY_PAIRS:
    if _pair in BASE_PAIR_FAMILIES:
        PAIR_FAMILIES[_pair] = BASE_PAIR_FAMILIES[_pair]
    else:
        PAIR_FAMILIES[_pair] = (f"{_pair[0]}_json", f"{_pair[0]}_kv")
PAIR_BY_FAMILY: dict[str, tuple[str, str]] = {
    family: pair
    for pair, families in PAIR_FAMILIES.items()
    for family in families
}
ALL_KEYS = frozenset(key for pair in KEY_PAIRS for key in pair)

TRAIN_FAMILIES = (
    "person_json", "product_json", "person_kv", "product_kv",
    "person_yaml", "product_yaml", "arith_plain", "arith_json",
)
CROSS_FAMILIES = ("city_json", "city_kv", "city_yaml")
LEGACY_TRAIN_FAMILIES = (
    "person_json", "product_json", "person_kv", "product_kv",
    "arith_plain", "arith_json",
)
DEFAULT_COUNTS = {
    "person_json": 40, "product_json": 40, "person_kv": 40,
    "product_kv": 40, "person_yaml": 40, "product_yaml": 40,
    "city_json": 25, "city_kv": 25, "city_yaml": 25,
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


def canonical_answer(family: str | FamilySpec, answer: str, entity: str,
                     number: int) -> str:
    """Return a compact, validated form of ``answer`` when possible.

    JSON objects are emitted with ``separators=(",", ":")`` and KV lines with
    no spaces. The function validates first and returns the cleaned input
    unchanged when it is not a valid answer. YAML and plain answers are not
    rewritten.
    """

    spec = get_family(family)
    cleaned = _clean(answer)
    if not validate_answer(spec, cleaned, entity, number):
        return cleaned
    if spec.format == "json":
        expected_text, expected_number = _expected(spec, entity, number)
        payload = {spec.key1: expected_number if spec.arithmetic
                   else expected_text}
        if spec.key2 is not None:
            payload[spec.key2] = expected_number
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    if spec.format == "kv":
        expected_text, expected_number = _expected(spec, entity, number)
        return f"{spec.key1}={expected_text};{spec.key2}={expected_number}"
    return cleaned


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


def _coerce_key_pairs(value) -> tuple[tuple[str, str], ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        pairs = parse_key_pairs(value)
        if not pairs:
            raise ValueError("at least one key pair is required")
        return pairs
    pairs: list[tuple[str, str]] = []
    for item in value:
        if isinstance(item, str):
            parsed = parse_key_pairs(item)
            for pair in parsed:
                if pair not in pairs:
                    pairs.append(pair)
            continue
        pair = (str(item[0]).strip().lower(), str(item[1]).strip().lower())
        if pair not in KEY_PAIRS:
            raise ValueError(f"unknown key pair: {pair[0]}:{pair[1]}")
        if pair not in pairs:
            pairs.append(pair)
    if not pairs:
        raise ValueError("at least one key pair is required")
    return tuple(pairs)


def parse_key_pairs(value: str | None) -> tuple[tuple[str, str], ...]:
    """Parse ``--key-pairs``.

    Accepts ``all``, comma-separated ``key1:key2`` pairs, or a bare ``key1``
    that selects its canonical pair. ``/`` and ``=`` also separate the keys.
    """

    if value is None:
        return ()
    text = str(value).strip()
    if not text:
        return ()
    if text.lower() == "all":
        return KEY_PAIRS
    by_first = {pair[0]: pair for pair in KEY_PAIRS}
    picked: list[tuple[str, str]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        parts = re.split(r"[:/=]", item, maxsplit=1)
        if len(parts) == 2:
            pair = (parts[0].strip().lower(), parts[1].strip().lower())
            if pair not in KEY_PAIRS:
                raise ValueError(
                    f"unknown key pair in --key-pairs: {item!r}")
        else:
            key = item.lower()
            if key not in by_first:
                raise ValueError(
                    f"unknown key in --key-pairs: {item!r}")
            pair = by_first[key]
        if pair not in picked:
            picked.append(pair)
    if not picked:
        raise ValueError("--key-pairs did not contain a usable pair")
    return tuple(picked)


def parse_holdout_keys(value: str | None) -> tuple[str, ...]:
    """Parse a comma-separated list of key names for ``--holdout-keys``."""

    if value is None:
        return ()
    keys: list[str] = []
    for item in str(value).split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in ALL_KEYS:
            raise ValueError(f"unknown key in --holdout-keys: {key!r}")
        if key not in keys:
            keys.append(key)
    return tuple(keys)


def resolve_family_plan(key_pairs=None,
                        holdout_keys: str | Sequence[str] = ()
                        ) -> dict:
    """Resolve selected key pairs and held-out keys to a family plan.

    Returns ``train_families``, ``cross_families`` (eval-only),
    ``key_pairs`` and ``holdout_keys``. Held-out keys always produce
    eval-only families; the original city families are also eval-only.
    """

    if holdout_keys is None:
        holds = ()
    elif isinstance(holdout_keys, str):
        holds = parse_holdout_keys(holdout_keys)
    else:
        holds = tuple(str(key).strip().lower() for key in holdout_keys
                      if str(key).strip())
    unknown = sorted(set(holds) - ALL_KEYS)
    if unknown:
        raise ValueError(f"unknown holdout keys: {', '.join(unknown)}")
    selected = _coerce_key_pairs(key_pairs)

    if selected is None:
        train = list(TRAIN_FAMILIES)
        cross = list(CROSS_FAMILIES)
        selected_pairs = list(KEY_PAIRS)
        for family in list(train):
            pair = PAIR_BY_FAMILY.get(family)
            if pair and set(pair) & set(holds):
                train.remove(family)
                cross.append(family)
    else:
        selected_pairs = list(selected)
        train = list(ARITH_FAMILIES)
        cross = []
        for pair in selected_pairs:
            eval_only = (pair == ("city", "population")
                         or bool(set(pair) & set(holds)))
            target = cross if eval_only else train
            for family in PAIR_FAMILIES[pair]:
                if family not in target:
                    target.append(family)

    train = _dedupe(train)
    cross = _dedupe(cross)
    overlap = set(train) & set(cross)
    if overlap:
        raise ValueError(f"train and eval-only families overlap: "
                         f"{sorted(overlap)}")
    missing = [family for family in train + cross if family not in FAMILIES]
    if missing:
        raise ValueError(f"unknown families in plan: {missing}")
    planned = set(train) | set(cross)
    generated_pairs = [
        pair for pair in selected_pairs
        if any(family in planned for family in PAIR_FAMILIES[pair])
    ]
    return {
        "train_families": train,
        "cross_families": cross,
        "key_pairs": tuple(generated_pairs),
        "holdout_keys": tuple(sorted(holds)),
    }


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _pool_entities(families: Iterable[str]) -> dict[str, tuple[str, ...]]:
    pools: dict[str, tuple[str, ...]] = {}
    for family in families:
        spec = FAMILIES[family]
        if spec.pool is None:
            continue
        current = pools.get(spec.pool)
        if current is not None and tuple(current) != tuple(spec.entities):
            raise ValueError(
                f"entity pool {spec.pool!r} has inconsistent values")
        pools[spec.pool] = tuple(spec.entities)
    return pools


def plan_entity_split(families: Iterable[str], *,
                      train_families: Sequence[str],
                      cross_families: Sequence[str],
                      seed: int = 17,
                      train_ratio: float = 0.8) -> dict[str, dict]:
    """Assign whole entity pools to a train or eval side before generation.

    Pools used by at least one train family are split deterministically with
    ``train_ratio``. Pools used only by eval-only families go entirely to the
    eval side. The returned mapping is ``pool -> {train, eval, entities}``.
    """

    if not 0.0 <= train_ratio <= 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    family_list = list(families)
    train_set = set(train_families)
    pools = _pool_entities(family_list)
    train_pools = {
        FAMILIES[family].pool for family in train_set
        if family in FAMILIES and FAMILIES[family].pool is not None
    }
    plan: dict[str, dict] = {}
    for pool, entities in pools.items():
        values = list(entities)
        if pool in train_pools and len(values) > 1:
            rng = random.Random(f"{seed}:{pool}")
            rng.shuffle(values)
            cut = max(1, int(round(len(values) * train_ratio)))
            cut = min(cut, len(values) - 1)
            plan[pool] = {
                "train": tuple(values[:cut]),
                "eval": tuple(values[cut:]),
                "entities": tuple(entities),
            }
        elif pool in train_pools:
            plan[pool] = {
                "train": tuple(values),
                "eval": (),
                "entities": tuple(entities),
            }
        else:
            plan[pool] = {
                "train": (),
                "eval": tuple(values),
                "entities": tuple(entities),
            }
    return plan


def _split_count(count: int, train_ratio: float) -> int:
    if count <= 0:
        return 0
    cut = max(1, int(round(count * train_ratio)))
    if count > 1:
        cut = min(cut, count - 1)
    return cut


def build_rows(families: Iterable[str],
               teacher_call: Callable[[str], str] | None = None, *,
               count: int | None = None,
               counts: Mapping[str, int] | None = None,
               seed: int = 17,
               progress: Callable[[dict], None] | None = None,
               split_by: str = "row",
               train_families: Sequence[str] | None = None,
               cross_families: Sequence[str] | None = None,
               train_ratio: float = 0.8,
               canonical: bool = False) -> list[dict]:
    """Generate and validate rows for ``families``.

    ``teacher_call`` receives the prompt and returns raw text. Pass ``None`` to
    use the deterministic oracle formatter, which is useful for tests and
    ``--dry-run``. No model is loaded here.

    With ``split_by="entity"`` rows are emitted after the entity pools have
    been split and carry a ``partition`` of ``train``, ``indist`` or ``cross``.
    Pass the same ``train_families``/``cross_families`` to :func:`split_rows`.
    """

    if split_by not in {"row", "entity"}:
        raise ValueError(f"unknown split mode: {split_by}")
    family_list = [families] if isinstance(families, str) else list(families)
    row_counts = _resolve_row_counts(family_list, counts, count)
    if split_by == "entity":
        cross_list = list(cross_families if cross_families is not None
                          else CROSS_FAMILIES)
        train_list = list(train_families if train_families is not None
                          else [family for family in family_list
                                if family not in set(cross_list)])
        overlap = set(train_list) & set(cross_list)
        if overlap:
            raise ValueError(
                f"train and eval-only families overlap: {sorted(overlap)}")
        known = set(train_list) | set(cross_list)
        unknown = [family for family in family_list if family not in known]
        if unknown:
            raise ValueError(
                f"families missing from the entity split: {unknown}")
        entity_plan = plan_entity_split(
            family_list, train_families=train_list,
            cross_families=cross_list, seed=seed, train_ratio=train_ratio)
    else:
        train_list = []
        cross_list = []
        entity_plan = None
    train_set = set(train_list)
    cross_set = set(cross_list)
    rng = random.Random(seed)
    rows: list[dict] = []
    total = 0
    accepted = 0

    def add_row(family: str, spec: FamilySpec, entity: str | None,
                number: int, partition: str | None) -> None:
        nonlocal total, accepted
        prompt = spec.prompt_builder(spec, entity if entity is not None
                                     else "", number)
        answer = (oracle_answer(spec, entity, number)
                  if teacher_call is None else teacher_call(prompt))
        answer = _clean(answer)
        verified = validate_answer(spec, answer, entity, number)
        if canonical and verified:
            answer = canonical_answer(spec, answer, entity, number)
        row = {
            "user": prompt,
            "assistant": answer,
            "family": family,
            "entity": None if spec.arithmetic else entity,
            "entity_pool": None if spec.arithmetic else spec.pool,
            "verification": {
                "verified": verified,
                "promotable": verified,
                "status": "verified" if verified else "unverified",
                "method": VERIFICATION_METHOD,
            },
        }
        if partition is not None:
            row["partition"] = partition
        rows.append(row)
        total += 1
        accepted += int(verified)
        if progress is not None and total % 20 == 0:
            progress({
                "generated": total, "accepted": accepted,
                "rejected": total - accepted, "family": family,
            })

    for family in family_list:
        spec = FAMILIES[family]
        family_count = row_counts[family]
        if split_by == "entity" and spec.pool is not None:
            pool_plan = entity_plan[spec.pool]
            if family in train_set:
                train_entities = pool_plan["train"]
                eval_entities = pool_plan["eval"]
                if not train_entities:
                    train_count = 0
                elif not eval_entities:
                    train_count = family_count
                else:
                    train_count = _split_count(family_count, train_ratio)
                eval_count = family_count - train_count
                for index in range(train_count):
                    entity = train_entities[index % len(train_entities)]
                    add_row(family, spec, entity,
                            rng.randint(*spec.number_range), "train")
                for index in range(eval_count):
                    entity = eval_entities[index % len(eval_entities)]
                    add_row(family, spec, entity,
                            rng.randint(*spec.number_range), "indist")
            else:
                eval_entities = pool_plan["eval"]
                if family_count and not eval_entities:
                    raise ValueError(
                        f"eval-only family {family} has no eval entities")
                for index in range(family_count):
                    entity = eval_entities[index % len(eval_entities)]
                    add_row(family, spec, entity,
                            rng.randint(*spec.number_range), "cross")
            continue

        if split_by == "entity" and family in train_set:
            train_count = _split_count(family_count, train_ratio)
            for index in range(family_count):
                entity, number = _case(spec, index, rng)
                partition = "train" if index < train_count else "indist"
                add_row(family, spec, entity, number, partition)
            continue

        if split_by == "entity" and family in cross_set:
            for index in range(family_count):
                entity, number = _case(spec, index, rng)
                add_row(family, spec, entity, number, "cross")
            continue

        for index in range(family_count):
            entity, number = _case(spec, index, rng)
            add_row(family, spec, entity, number, None)
    return rows


def verified_rows(rows: Iterable[dict]) -> list[dict]:
    """Return only rows whose deterministic validator accepted the answer."""

    return [row for row in rows if row["verification"]["verified"]]


def extract_replay(rows: Iterable[dict],
                   replay_families: Sequence[str] = LEGACY_TRAIN_FAMILIES
                   ) -> tuple[list[dict], list[dict]]:
    """Split train rows into ``(primary, replay)`` without duplication."""

    replay_set = set(replay_families)
    primary: list[dict] = []
    replay: list[dict] = []
    for row in rows:
        (replay if row["family"] in replay_set else primary).append(row)
    return primary, replay


def label_replay_rows(rows: Iterable[dict]) -> list[dict]:
    """Return copies marked as replay-only training rows."""

    labeled: list[dict] = []
    for row in rows:
        item = dict(row)
        item["replay"] = True
        item["partition"] = "replay"
        labeled.append(item)
    return labeled


def shuffle_rows(rows: Iterable[dict], *, seed: int = 17) -> list[dict]:
    """Deterministically shuffle rows with ``seed``."""

    result = list(rows)
    random.Random(f"{seed}:shuffle").shuffle(result)
    return result


def split_rows(rows: Sequence[dict], *,
               train_families: Sequence[str] = TRAIN_FAMILIES,
               cross_families: Sequence[str] = CROSS_FAMILIES,
               train_ratio: float = 0.8,
               split_by: str = "row") -> dict[str, list[dict]]:
    """Split verified rows into train/indist/cross/all without family leaks.

    ``split_by="row"`` keeps the original per-family 80/20 slice. With
    ``split_by="entity"`` the ``partition`` assigned by :func:`build_rows`
    decides the side, which keeps shared entity pools disjoint.
    """

    if split_by not in {"row", "entity"}:
        raise ValueError(f"unknown split mode: {split_by}")
    overlap = set(train_families) & set(cross_families)
    if overlap:
        raise ValueError(
            f"train and cross families overlap: {sorted(overlap)}")
    if not 0.0 <= train_ratio <= 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    verified = verified_rows(rows)
    train_set = set(train_families)
    cross_set = set(cross_families)
    if split_by == "entity":
        train: list[dict] = []
        indist: list[dict] = []
        cross: list[dict] = []
        for row in verified:
            family = row["family"]
            if family in train_set:
                partition = row.get("partition")
                if partition == "train":
                    train.append(row)
                elif partition == "indist":
                    indist.append(row)
                else:
                    raise ValueError(
                        f"train family {family} has no entity partition")
            elif family in cross_set:
                cross.append(row)
            else:
                raise ValueError(
                    f"family {family} is neither train nor eval-only")
        return {
            "train": train, "indist": indist, "cross": cross,
            "all": verified,
        }

    train = []
    indist = []
    for family in train_families:
        subset = [row for row in verified if row["family"] == family]
        if not subset:
            continue
        cut = max(1, int(round(len(subset) * train_ratio)))
        train.extend(subset[:cut])
        indist.extend(subset[cut:])
    cross = [row for row in verified if row["family"] in cross_set]
    return {"train": train, "indist": indist, "cross": cross, "all": verified}


def entity_keys(rows: Iterable[dict]) -> set[tuple[str, str]]:
    """Return unique ``(pool, entity)`` keys for non-arithmetic rows."""

    keys: set[tuple[str, str]] = set()
    for row in rows:
        pool = row.get("entity_pool")
        entity = row.get("entity")
        if pool and entity is not None:
            keys.add((str(pool), str(entity)))
    return keys


def split_summary(files: Mapping[str, Sequence[dict]], *,
                  mode: str,
                  replay_rows: Sequence[dict] = ()) -> dict:
    """Build a machine-readable entity/train split summary."""

    train_entities = entity_keys(files.get("train", ()))
    eval_entities = entity_keys(
        list(files.get("indist", ())) + list(files.get("cross", ())))
    replay_entities = entity_keys(replay_rows)
    training_entities = train_entities | replay_entities
    overlap = training_entities & eval_entities
    return {
        "mode": mode,
        "entities_train": len(training_entities),
        "entities_eval": len(eval_entities),
        "entities_primary_train": len(train_entities),
        "entities_replay": len(replay_entities),
        "overlap": len(overlap),
        "entity_overlap": len(overlap),
        "entity_overlap_examples": sorted(
            f"{pool}:{entity}" for pool, entity in overlap)[:10],
    }


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


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8")
    return target


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


def _row_key(row: Mapping) -> tuple[str, str, str]:
    return (str(row.get("family")), str(row.get("user")),
            str(row.get("assistant")))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=r"C:\Users\biris\Desktop\opencode\ai gelistir\models"
                r"\Qwen2.5-1.5B-Instruct")
    parser.add_argument("--out-dir", default="data/diverse_teacher")
    parser.add_argument(
        "--per-family", type=int, default=None,
        help="rows per generated family; default 20 with --key-pairs or "
             "--holdout-keys, otherwise 40 (legacy default)")
    parser.add_argument(
        "--city-per-family", type=int, default=None,
        help="city_json/city_kv rows; defaults to --per-family in curriculum "
             "mode, otherwise 25")
    parser.add_argument(
        "--city-yaml-count", type=int, default=None,
        help="city_yaml rows; defaults to --city-per-family")
    parser.add_argument(
        "--arith-count", type=int, default=None,
        help="rows per arithmetic family; defaults to --per-family in "
             "curriculum mode, otherwise 20")
    parser.add_argument(
        "--family-count", action="append", default=[], type=_family_count_arg,
        metavar="FAMILY=COUNT", help="override one family count; repeatable")
    parser.add_argument(
        "--key-pairs", default=None,
        help="comma list of key pairs (key1:key2 or key1) or 'all'; "
             "default keeps the original families")
    parser.add_argument(
        "--holdout-keys", default=None,
        help="comma list of key names; families containing them are "
             "eval-only and never enter train")
    parser.add_argument(
        "--canonical-json", action="store_true",
        help="re-emit verified JSON compactly and KV without spaces")
    parser.add_argument(
        "--split-by", choices=("row", "entity"), default="row",
        help="row slice (legacy) or entity-first train/eval split")
    parser.add_argument(
        "--shuffle", action="store_true",
        help="deterministically shuffle generated rows with --seed")
    parser.add_argument(
        "--replay-out", default=None, metavar="PATH",
        help="write legacy-family training rows to PATH and remove them "
             "from train.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="use the deterministic oracle formatter, no model load")
    args = parser.parse_args(argv)

    try:
        plan = resolve_family_plan(args.key_pairs, args.holdout_keys)
    except ValueError as exc:
        parser.error(str(exc))

    overrides = dict(args.family_count)
    unknown = sorted(set(overrides) - set(FAMILIES))
    if unknown:
        parser.error(f"unknown family in --family-count: {', '.join(unknown)}")
    curriculum_mode = args.key_pairs is not None or args.holdout_keys
    per_family = (args.per_family if args.per_family is not None
                  else (20 if curriculum_mode else 40))
    if args.city_per_family is not None:
        city_per_family = args.city_per_family
    else:
        city_per_family = per_family if curriculum_mode else 25
    if args.arith_count is not None:
        arith_count = args.arith_count
    else:
        arith_count = per_family if curriculum_mode else 20
    family_order = plan["train_families"] + plan["cross_families"]
    try:
        counts = resolve_counts(
            family_order,
            per_family=per_family,
            city_per_family=city_per_family,
            city_yaml_count=args.city_yaml_count,
            arith_count=arith_count,
            overrides=overrides,
        )
    except ValueError as exc:
        parser.error(str(exc))

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
        family_order, teacher_call, counts=counts, seed=args.seed,
        progress=progress, split_by=args.split_by,
        train_families=plan["train_families"],
        cross_families=plan["cross_families"],
        canonical=args.canonical_json)
    if args.shuffle:
        rows = shuffle_rows(rows, seed=args.seed)
    files = split_rows(
        rows, train_families=plan["train_families"],
        cross_families=plan["cross_families"], split_by=args.split_by)

    out_dir = Path(args.out_dir).expanduser()
    replay_rows: list[dict] = []
    replay_path: Path | None = None
    if args.replay_out:
        primary, replay = extract_replay(files["train"])
        if not primary:
            parser.error(
                "--replay-out would leave train.jsonl empty; select at least "
                "one non-legacy key pair with --key-pairs")
        files["train"] = primary
        replay_rows = label_replay_rows(replay)
        replay_path = Path(args.replay_out).expanduser()
        for name in files:
            if replay_path.resolve() == (out_dir / f"{name}.jsonl").resolve():
                parser.error(
                    f"--replay-out must not overwrite {name}.jsonl")

    report = split_summary(files, mode=args.split_by,
                           replay_rows=replay_rows)
    if args.split_by == "entity" and report["entity_overlap"]:
        parser.error(
            "entity split produced overlapping train/eval entities: "
            f"{report['entity_overlap_examples']}")

    written = write_rows(files, out_dir)
    if replay_path is not None:
        write_jsonl(replay_path, replay_rows)
    verified = files["all"]
    train_keys = {_row_key(row) for row in files["train"]}
    replay_keys = {_row_key(row) for row in replay_rows}
    print(json.dumps({
        "requested": len(rows),
        "accepted": len(verified),
        "rejected": len(rows) - len(verified),
        "train": len(files["train"]),
        "indist": len(files["indist"]),
        "cross": len(files["cross"]),
        "replay": len(replay_rows),
        "replay_out": str(replay_path) if replay_path else None,
        "train_replay_duplicates": len(train_keys & replay_keys),
        "families": family_order,
        "eval_only_families": list(plan["cross_families"]),
        "key_pairs": [f"{key1}:{key2}"
                      for key1, key2 in plan["key_pairs"]],
        "holdout_keys": list(plan["holdout_keys"]),
        "canonical_json": bool(args.canonical_json),
        "shuffled": bool(args.shuffle),
        "split": report,
        "by_family": {
            family: sum(1 for row in verified if row["family"] == family)
            for family in family_order
        },
        "counts": counts,
        "out_dir": str(out_dir),
        "files": {name: str(path) for name, path in written.items()},
        "device": device,
        "teacher": teacher_name,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
