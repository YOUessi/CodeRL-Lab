#!/usr/bin/env bash
set -euo pipefail

PREFERENCES="${PREFERENCES:-artifacts/exp004b/random-label-control/preferences.jsonl}"
SFT_ADAPTER="${SFT_ADAPTER:-artifacts/exp006a/sft-qwen3-1.7b-lora}"
OUTPUT="${OUTPUT:-artifacts/exp004b/dpo-random-label-qwen3-1.7b}"

test -f "$PREFERENCES"
test -f "$SFT_ADAPTER/adapter_model.safetensors"

python -m coderl_lab.train.dpo \
  --config configs/dpo_runtime_repair_qwen3_1.7b.yaml \
  --preferences "$PREFERENCES" \
  --sft-adapter "$SFT_ADAPTER" \
  --output-dir "$OUTPUT"
