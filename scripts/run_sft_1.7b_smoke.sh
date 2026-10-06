#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/generated/mbpp-v1}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/exp006a/smoke-sft-qwen3-1.7b-lora}"

if [ ! -f "$DATA_DIR/train_sft.jsonl" ]; then
  bash scripts/prepare_mbpp.sh
fi

python -m coderl_lab.train.sft \
  --config configs/sft_qwen3_1.7b.yaml \
  --data "$DATA_DIR/train_sft.jsonl" \
  --output-dir "$OUTPUT_DIR" \
  --max-samples 32 \
  --num-train-epochs 1
