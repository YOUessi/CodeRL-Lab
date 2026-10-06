#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/generated/mbpp-v1}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/exp002/sft-qwen3-0.6b-lora}"

if [ ! -f "$DATA_DIR/train_sft.jsonl" ]; then
  echo "MBPP-v1 not found; preparing pinned dataset."
  bash scripts/prepare_mbpp.sh
fi

python -m coderl_lab.train.sft \
  --config configs/sft_qwen3_0.6b.yaml \
  --data "$DATA_DIR/train_sft.jsonl" \
  --output-dir "$OUTPUT_DIR"
