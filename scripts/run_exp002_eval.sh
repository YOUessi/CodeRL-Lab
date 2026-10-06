#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/generated/mbpp-v1}"
ROOT="${ROOT:-artifacts/exp002/eval-validation}"
MODEL="Qwen/Qwen3-0.6B-Base"
REVISION="da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
ADAPTER="${ADAPTER:-artifacts/exp002/sft-qwen3-0.6b-lora}"

mkdir -p "$ROOT/base" "$ROOT/sft"

for MODE in base sft; do
  EXTRA=()
  if [ "$MODE" = "sft" ]; then
    EXTRA=(--adapter "$ADAPTER")
  fi

  python -m coderl_lab.generation \
    --tasks "$DATA_DIR/validation_tasks.jsonl" \
    --output "$ROOT/$MODE/predictions.jsonl" \
    --metadata-output "$ROOT/$MODE/generation.json" \
    --model "$MODEL" \
    --revision "$REVISION" \
    "${EXTRA[@]}" \
    --batch-samples \
    --num-samples 4 \
    --max-new-tokens 512 \
    --temperature 0.8 \
    --top-p 0.95 \
    --seed 42

  python -m coderl_lab.evaluation \
    --tasks "$DATA_DIR/validation_tasks.jsonl" \
    --predictions "$ROOT/$MODE/predictions.jsonl" \
    --output "$ROOT/$MODE/evaluation" \
    --executor docker \
    --timeout 5 \
    --memory 512m \
    --k 1 4
done
