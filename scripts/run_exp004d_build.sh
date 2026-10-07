#!/usr/bin/env bash
set -euo pipefail

SFT_ADAPTER="${SFT_ADAPTER:-artifacts/exp006a/sft-qwen3-1.7b-lora}"
REFERENCE_ADAPTER="${REFERENCE_ADAPTER:-artifacts/exp004c/dpo-noop-qwen3-1.7b}"
ROOT="${ROOT:-artifacts/exp004d}"

test -f "$SFT_ADAPTER/adapter_model.safetensors"
test -f "$REFERENCE_ADAPTER/adapter_model.safetensors"

for SEED in 101 202 303; do
  OUT="$ROOT/random-$SEED"
  rm -rf "$OUT"
  python -m coderl_lab.analysis.matched_random_perturbation \
    --base-adapter "$SFT_ADAPTER/adapter_model.safetensors" \
    --reference-adapter "$REFERENCE_ADAPTER/adapter_model.safetensors" \
    --output-dir "$OUT" \
    --seed "$SEED"
done
