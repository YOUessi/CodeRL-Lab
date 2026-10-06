#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/generated/mbpp-v1}"
ROOT="${ROOT:-artifacts/exp003/eval-validation}"
MODEL="Qwen/Qwen3-0.6B-Base"
REVISION="da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
GRPO_ADAPTER="${GRPO_ADAPTER:-artifacts/exp003/grpo-qwen3-0.6b}"

if [ ! -f "$GRPO_ADAPTER/adapter_model.safetensors" ]; then
  echo "Missing GRPO adapter: $GRPO_ADAPTER" >&2
  exit 2
fi

mkdir -p "$ROOT/grpo"

python -m coderl_lab.generation \
  --tasks "$DATA_DIR/validation_tasks.jsonl" \
  --output "$ROOT/grpo/predictions.jsonl" \
  --metadata-output "$ROOT/grpo/generation.json" \
  --model "$MODEL" \
  --revision "$REVISION" \
  --adapter "$GRPO_ADAPTER" \
  --batch-samples \
  --num-samples 4 \
  --max-new-tokens 512 \
  --temperature 0.8 \
  --top-p 0.95 \
  --seed 42

python -m coderl_lab.evaluation \
  --tasks "$DATA_DIR/validation_tasks.jsonl" \
  --predictions "$ROOT/grpo/predictions.jsonl" \
  --output "$ROOT/grpo/evaluation" \
  --executor docker \
  --timeout 5 \
  --memory 512m \
  --k 1 4
