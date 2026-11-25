#!/usr/bin/env bash
# Vast.ai experiment runner for the 0.90.0 capacity scaling question.
#
# Expected layout on the instance:
#   /workspace/atomzero-bundle.tar.gz
#   /workspace/out
#
# The script downloads the public 49M base from Hugging Face, upcycles it to
# N layers, fully fine-tunes on copy data, evaluates held-out 2/3/4-key copy
# and content retention, and writes a content-free summary. It never uploads
# anything unless HF_TOKEN and HF_REPO are explicitly set.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
OUT="${OUT:-$WORKSPACE/out}"
LAYERS="${LAYERS:-24}"
SEEDS="${SEEDS:-101}"
STEPS="${STEPS:-4000}"
BATCH="${BATCH:-16}"
LR="${LR:-0.00005}"
CONTENT_WEIGHT="${CONTENT_WEIGHT:-2}"
EVAL_ROWS="${EVAL_ROWS:-200}"
RUN_RETENTION="${RUN_RETENTION:-1}"
BASE_REPO="${BASE_REPO:-byjameson/atom-zero-49m}"
TRAIN_DATA="${TRAIN_DATA:-data/multi_short_v1/train_all_plus_distill.jsonl}"
CACHE_DIR="${CACHE_DIR:-}"
RUN_TAG="${RUN_TAG:-fullft_${LAYERS}l}"
PY="${PYTHON:-python3}"

mkdir -p "$OUT"
exec > >(tee -a "$OUT/job.log") 2>&1
echo "=== atomzero vastai job $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "layers=$LAYERS seeds=$SEEDS steps=$STEPS batch=$BATCH lr=$LR"
nvidia-smi || true

fail() {
  echo "FAILED: $*" >&2
  echo "$*" > "$OUT/FAILED"
  exit 1
}

cd "$WORKSPACE"
if [ ! -f atomzero/pyproject.toml ]; then
  mkdir -p atomzero
  tar -xzf atomzero-bundle.tar.gz -C atomzero
fi
cd "$WORKSPACE/atomzero" || fail "bundle did not unpack to /workspace/atomzero"
export PYTHONPATH="$PWD:$PWD/src"
export PYTHONIOENCODING=utf-8

"$PY" -m pip install -q -e . || fail "pip install failed"

BASE_DIR="$WORKSPACE/checkpoints/base_49m"
if [ ! -f "$BASE_DIR/model.pt" ]; then
  mkdir -p "$BASE_DIR"
  for f in model.pt config.json chat_en_bpe8k.json; do
    "$PY" - "$BASE_DIR/$f" "https://huggingface.co/$BASE_REPO/resolve/main/$f" <<'PY' || fail "download $f failed"
import sys, urllib.request
urllib.request.urlretrieve(sys.argv[2], sys.argv[1])
print("downloaded", sys.argv[1])
PY
  done
fi
echo "base checkpoint ready: $(ls -lh "$BASE_DIR" | wc -l) files"

UP_DIR="$WORKSPACE/checkpoints/upcycle_${LAYERS}l"
"$PY" research/upcycle_model.py \
  --base "$BASE_DIR" \
  --out "$UP_DIR" \
  --layers "$LAYERS" \
  --mode interpolate \
  --seed 23 \
  --report "$OUT/upcycle_${LAYERS}l.json" || fail "upcycle failed"

SUMMARY_ARGS=()
for SEED in $SEEDS; do
  NAME="${RUN_TAG}_s${SEED}"
  CKPT="$WORKSPACE/checkpoints/$NAME"
  echo "=== train $NAME on $TRAIN_DATA cache=${CACHE_DIR:-none} ==="
  "$PY" research/finetune_copy_control.py \
    --base "$UP_DIR" \
    --train "$TRAIN_DATA" \
    --cache-dir "$CACHE_DIR" \
    --eval data/staged_copy_v1/mixed_eval.jsonl \
    --out "$CKPT" \
    --steps "$STEPS" \
    --lr "$LR" \
    --batch-size "$BATCH" \
    --max-seq 256 \
    --content-weight "$CONTENT_WEIGHT" \
    --seed "$SEED" \
    --result "$OUT/${NAME}.json" \
    --comparison "$OUT/${NAME}_cmp.json" || fail "training $NAME failed"

  for STAGE in 2short 3short 4short; do
    "$PY" research/compare_adapters.py \
      --base "$CKPT" \
      --data "data/multi_short_v1/stage_${STAGE}/eval.jsonl" \
      --exact-match --key-copy --device cuda \
      --out "$OUT/${NAME}_${STAGE}.json" || fail "eval $NAME $STAGE failed"
  done
  SUMMARY_ARGS+=("$NAME")

  if [ "$RUN_RETENTION" = "1" ]; then
    "$PY" research/retention_eval.py \
      --baseline "$BASE_DIR" \
      --candidate "$CKPT" \
      --data data/eval_frozen/heldout_city_yaml.jsonl \
             data/eval_frozen/heldout_city_river.jsonl \
      --max-rows 200 --per-family --device cuda \
      --out "$OUT/${NAME}_retention.json" || echo "retention failed for $NAME"
  fi

  if [ -n "${HF_TOKEN:-}" ] && [ -n "${HF_REPO:-}" ]; then
    "$PY" - "$CKPT" "$HF_REPO" <<'PY' || echo "HF upload failed"
import os, sys
from huggingface_hub import HfApi
ckpt, repo = sys.argv[1], sys.argv[2]
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=ckpt, repo_id=repo, repo_type="model",
                  commit_message="0.90.0 capacity run")
print("uploaded", repo)
PY
  fi
done

"$PY" - "$OUT" "$LAYERS" "${SUMMARY_ARGS[@]}" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
layers = int(sys.argv[2])
names = sys.argv[3:]
summary = {"schema": "atomzero.vastai_run.v1", "layers": layers, "runs": {}}
for name in names:
    run = {"stages": {}, "retention": None}
    for stage in ("2short", "3short", "4short"):
        p = out / f"{name}_{stage}.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        model = data["models"][0]["metrics"]
        run["stages"][stage] = {
            "examples": model.get("examples"),
            "exact": model.get("exact_match", {}).get("rate"),
            "key_copy": model.get("key_copy", {}).get("rate"),
            "loss": round(float(model.get("loss", 0.0)), 6),
        }
    p = out / f"{name}_retention.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        run["retention"] = {
            "retention_ratio": data.get("retention_ratio"),
            "retention_ok": data.get("retention_ok"),
        }
    train = out / f"{name}.json"
    if train.exists():
        t = json.loads(train.read_text(encoding="utf-8"))
        run["train_loss"] = t.get("train_loss")
        run["params"] = t.get("total_parameters", t.get("trainable_parameters"))
    summary["runs"][name] = run
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/DONE"
echo "=== finished ==="
