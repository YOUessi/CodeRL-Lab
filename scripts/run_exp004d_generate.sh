#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-artifacts/exp006a/sft-qwen3-1.7b-lora}"
REFERENCE="${REFERENCE:-artifacts/exp004c/dpo-noop-qwen3-1.7b}"
ROOT="${ROOT:-artifacts/exp004d/perturbations}"

mkdir -p "$ROOT"

for SEED in 101 202 303; do
  OUT="$ROOT/seed-$SEED"
  rm -rf "$OUT"
  python -m coderl_lab.analysis.matched_norm_perturb \
    --base-adapter "$BASE" \
    --reference-adapter "$REFERENCE" \
    --output-dir "$OUT" \
    --seed "$SEED"
done

python -m coderl_lab.analysis.adapter_delta_cosine \
  --base "$BASE/adapter_model.safetensors" \
  --arm "noop05=$REFERENCE/adapter_model.safetensors" \
  --arm "seed101=$ROOT/seed-101/adapter_model.safetensors" \
  --arm "seed202=$ROOT/seed-202/adapter_model.safetensors" \
  --arm "seed303=$ROOT/seed-303/adapter_model.safetensors" \
  --output "$ROOT/delta_cosine.json"
