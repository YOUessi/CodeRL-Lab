#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/generated/mbpp-v1}"
SFT_ADAPTER="${SFT_ADAPTER:-artifacts/exp002/sft-qwen3-0.6b-lora}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/exp005b/smoke-online-qwen3-0.6b}"

if [ ! -f "$DATA_DIR/train_tasks.jsonl" ]; then
  bash scripts/prepare_mbpp.sh
fi
if [ ! -f "$SFT_ADAPTER/adapter_model.safetensors" ]; then
  echo "Missing EXP-002 SFT adapter: $SFT_ADAPTER" >&2
  exit 2
fi

docker image inspect python:3.11-slim >/dev/null 2>&1 || docker pull python:3.11-slim

python -m coderl_lab.train.grpo \
  --config configs/grpo_online_dynamic_qwen3_0.6b.yaml \
  --tasks "$DATA_DIR/train_tasks.jsonl" \
  --sft-adapter "$SFT_ADAPTER" \
  --output-dir "$OUTPUT_DIR" \
  --max-tasks 32 \
  --max-steps 4

python -m coderl_lab.analysis.grpo_history \
  --input "$OUTPUT_DIR/log_history.json" \
  --output "$OUTPUT_DIR/dynamics_summary.json"
