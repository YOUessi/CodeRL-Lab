#!/usr/bin/env bash
set -euo pipefail

ARM="${ARM:?Set ARM, e.g. random_control or boundary}"
ADAPTER="${ADAPTER:?Set ADAPTER path}"
DATA_DIR="${DATA_DIR:-data/generated/mbpp-v1}"
ROOT="${ROOT:-artifacts/exp005/eval-validation}"
MODEL="Qwen/Qwen3-0.6B-Base"
REVISION="da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
OUT="$ROOT/$ARM"

if [ ! -f "$ADAPTER/adapter_model.safetensors" ]; then
  echo "Missing adapter: $ADAPTER" >&2
  exit 2
fi

rm -rf "$OUT"
mkdir -p "$OUT"

python -m coderl_lab.generation \
  --tasks "$DATA_DIR/validation_tasks.jsonl" \
  --output "$OUT/predictions.jsonl" \
  --metadata-output "$OUT/generation.json" \
  --model "$MODEL" \
  --revision "$REVISION" \
  --adapter "$ADAPTER" \
  --batch-samples \
  --num-samples 4 \
  --max-new-tokens 512 \
  --temperature 0.8 \
  --top-p 0.95 \
  --seed 42

python -m coderl_lab.evaluation \
  --tasks "$DATA_DIR/validation_tasks.jsonl" \
  --predictions "$OUT/predictions.jsonl" \
  --output "$OUT/evaluation" \
  --executor docker \
  --timeout 5 \
  --memory 512m \
  --max-workers 4 \
  --k 1 4

python -m coderl_lab.analysis.evaluation_summary \
  --details "$OUT/evaluation/details.jsonl" \
  --summary "$OUT/evaluation/summary.json" \
  --output "$OUT/evaluation/enhanced_summary.json"
