#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/generated/mbpp-v1}"
ROOT="${ROOT:-artifacts/exp006a/eval-validation}"
MODEL="Qwen/Qwen3-1.7B-Base"
REVISION="ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
ADAPTER="${ADAPTER:-artifacts/exp006a/grpo-qwen3-1.7b}"
OUT="$ROOT/grpo"

if [ ! -f "$ADAPTER/adapter_model.safetensors" ]; then
  echo "Missing EXP-006A GRPO adapter: $ADAPTER" >&2
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
