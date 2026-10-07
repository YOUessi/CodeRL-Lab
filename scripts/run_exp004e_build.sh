#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-artifacts/exp006a/sft-qwen3-1.7b-lora}"
REFERENCE="${REFERENCE:-artifacts/exp004c/dpo-noop-qwen3-1.7b}"
ROOT="${ROOT:-artifacts/exp004e/perturbations}"

mkdir -p "$ROOT"

build_scale () {
  local label="$1"
  local scale="$2"
  for seed in 101 202 303; do
    local out="$ROOT/scale-$label/seed-$seed"
    rm -rf "$out"
    python -m coderl_lab.analysis.matched_norm_perturb \
      --base-adapter "$BASE" \
      --reference-adapter "$REFERENCE" \
      --output-dir "$out" \
      --seed "$seed" \
      --scale "$scale"
  done
}

build_scale 025 0.25
build_scale 050 0.50
build_scale 200 2.00

for seed in 101 202 303; do
  python -m coderl_lab.analysis.adapter_delta_cosine \
    --base "$BASE/adapter_model.safetensors" \
    --arm "scale025=$ROOT/scale-025/seed-$seed/adapter_model.safetensors" \
    --arm "scale050=$ROOT/scale-050/seed-$seed/adapter_model.safetensors" \
    --arm "scale200=$ROOT/scale-200/seed-$seed/adapter_model.safetensors" \
    --output "$ROOT/direction_cosine_seed_$seed.json"
done
