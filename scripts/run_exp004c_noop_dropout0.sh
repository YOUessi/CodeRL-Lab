#!/usr/bin/env bash
set -euo pipefail

PREFERENCES="${PREFERENCES:-artifacts/exp004c/controls/noop/preferences.jsonl}"
SFT="${SFT:-artifacts/exp006a/sft-qwen3-1.7b-lora}"
OUT="${OUT:-artifacts/exp004c/dpo-noop-dropout0-qwen3-1.7b}"

python -m coderl_lab.train.dpo \
  --config configs/dpo_deconcentration_noop_dropout0_qwen3_1.7b.yaml \
  --preferences "$PREFERENCES" \
  --sft-adapter "$SFT" \
  --output-dir "$OUT"

python -m coderl_lab.analysis.adapter_delta \
  --a "$SFT/adapter_model.safetensors" \
  --b "$OUT/adapter_model.safetensors" \
  --output "$OUT/adapter_delta_vs_sft.json"
