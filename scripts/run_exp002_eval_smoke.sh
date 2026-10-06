#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/generated/mbpp-v1}"
ROOT="${ROOT:-artifacts/exp002/eval-smoke}"
MODEL="Qwen/Qwen3-0.6B-Base"
REVISION="da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
ADAPTER="${ADAPTER:-artifacts/exp002/smoke-qwen3-0.6b-lora}"

mkdir -p "$ROOT/base" "$ROOT/sft"

python -m coderl_lab.generation \
  --tasks "$DATA_DIR/validation_tasks.jsonl" \
  --output "$ROOT/base/predictions.jsonl" \
  --metadata-output "$ROOT/base/generation.json" \
  --model "$MODEL" \
  --revision "$REVISION" \
  --max-tasks 20 \
  --num-samples 4 \
  --max-new-tokens 512 \
  --temperature 0.8 \
  --top-p 0.95 \
  --seed 42

python -m coderl_lab.evaluation \
  --tasks "$DATA_DIR/validation_tasks.jsonl" \
  --predictions "$ROOT/base/predictions.jsonl" \
  --output "$ROOT/base/evaluation" \
  --executor docker \
  --timeout 5 \
  --memory 512m \
  --k 1 4

python -m coderl_lab.generation \
  --tasks "$DATA_DIR/validation_tasks.jsonl" \
  --output "$ROOT/sft/predictions.jsonl" \
  --metadata-output "$ROOT/sft/generation.json" \
  --model "$MODEL" \
  --revision "$REVISION" \
  --adapter "$ADAPTER" \
  --max-tasks 20 \
  --num-samples 4 \
  --max-new-tokens 512 \
  --temperature 0.8 \
  --top-p 0.95 \
  --seed 42

python -m coderl_lab.evaluation \
  --tasks "$DATA_DIR/validation_tasks.jsonl" \
  --predictions "$ROOT/sft/predictions.jsonl" \
  --output "$ROOT/sft/evaluation" \
  --executor docker \
  --timeout 5 \
  --memory 512m \
  --k 1 4
