#!/usr/bin/env bash
set -euo pipefail

SCREEN_DIR="${SCREEN_DIR:-artifacts/exp005/screen-v1}"
TASKS="${TASKS:-$SCREEN_DIR/random_control_train_tasks.jsonl}"
SFT_ADAPTER="${SFT_ADAPTER:-artifacts/exp002/sft-qwen3-0.6b-lora}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/exp005/grpo-random-subset-qwen3-0.6b}"

if [ ! -f "$TASKS" ]; then
  echo "Random control task file not found; run boundary screening first." >&2
  exit 2
fi

docker image inspect python:3.11-slim >/dev/null 2>&1 || docker pull python:3.11-slim

python -m coderl_lab.train.grpo \
  --config configs/grpo_random_subset_qwen3_0.6b.yaml \
  --tasks "$TASKS" \
  --sft-adapter "$SFT_ADAPTER" \
  --output-dir "$OUTPUT_DIR" \
  --max-steps 187

python -m coderl_lab.analysis.grpo_history \
  --input "$OUTPUT_DIR/log_history.json" \
  --output "$OUTPUT_DIR/dynamics_summary.json"
