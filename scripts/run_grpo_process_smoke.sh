#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/generated/mbpp-v1}"
SFT_ADAPTER="${SFT_ADAPTER:-artifacts/exp006a/sft-qwen3-1.7b-lora}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/exp004a/smoke-grpo-process-qwen3-1.7b}"

if [ ! -f "$DATA_DIR/train_tasks.jsonl" ]; then
  bash scripts/prepare_mbpp.sh
fi
if [ ! -f "$SFT_ADAPTER/adapter_model.safetensors" ]; then
  echo "Missing EXP-006A SFT adapter: $SFT_ADAPTER" >&2
  exit 2
fi

docker image inspect python:3.11-slim >/dev/null 2>&1 || docker pull python:3.11-slim

python -m coderl_lab.train.grpo \
  --config configs/grpo_process_qwen3_1.7b.yaml \
  --tasks "$DATA_DIR/train_tasks.jsonl" \
  --sft-adapter "$SFT_ADAPTER" \
  --output-dir "$OUTPUT_DIR" \
  --max-tasks 16 \
  --max-steps 2

python -m coderl_lab.analysis.grpo_history \
  --input "$OUTPUT_DIR/log_history.json" \
  --output "$OUTPUT_DIR/dynamics_summary.json"
