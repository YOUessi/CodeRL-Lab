#!/usr/bin/env bash
set -euo pipefail

PREFS="${PREFS:-artifacts/exp004b/random-label-control/preferences.jsonl}"
SFT_ADAPTER="${SFT_ADAPTER:-artifacts/exp006a/sft-qwen3-1.7b-lora}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/exp004b/dpo-random-control-qwen3-1.7b}"

test -f "$PREFS"
test -f "$SFT_ADAPTER/adapter_model.safetensors"

python -m coderl_lab.train.dpo \
  --config configs/dpo_runtime_repair_random_control_qwen3_1.7b.yaml \
  --preferences "$PREFS" \
  --sft-adapter "$SFT_ADAPTER" \
  --output-dir "$OUTPUT_DIR"
