#!/usr/bin/env bash
set -euo pipefail

ARM="${ARM:?Set ARM}"
ADAPTER="${ADAPTER:?Set ADAPTER}"
ROOT="${ROOT:-artifacts/exp004c/eval-validation}"
DATA_DIR="${DATA_DIR:-data/generated/mbpp-v1}"
MODEL="Qwen/Qwen3-1.7B-Base"
REVISION="ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
OUT="$ROOT/$ARM"

test -f "$ADAPTER/adapter_model.safetensors"
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
  --sample-batch-size 16 \
  --num-samples 16 \
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
  --k 1 4 8 16

python -m coderl_lab.analysis.evaluation_summary \
  --details "$OUT/evaluation/details.jsonl" \
  --summary "$OUT/evaluation/summary.json" \
  --output "$OUT/evaluation/enhanced_summary.json"

python -m coderl_lab.analysis.completion_diversity \
  --predictions "$OUT/predictions.jsonl" \
  --output "$OUT/diversity.json"
