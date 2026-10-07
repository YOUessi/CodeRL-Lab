#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-artifacts/exp004f/temperature}"
DATA="${DATA:-data/generated/mbpp-v1/validation_tasks.jsonl}"
MODEL="Qwen/Qwen3-1.7B-Base"
REVISION="ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
SFT="${SFT:-artifacts/exp006a/sft-qwen3-1.7b-lora}"

mkdir -p "$ROOT"

run_policy () {
  local TEMP_TAG="$1"
  local TEMP_VALUE="$2"
  local ARM="$3"
  local ADAPTER="$4"
  local OUT="$ROOT/$TEMP_TAG/$ARM"

  rm -rf "$OUT"
  mkdir -p "$OUT"

  python -m coderl_lab.generation \
    --tasks "$DATA" \
    --output "$OUT/predictions.jsonl" \
    --metadata-output "$OUT/generation.json" \
    --model "$MODEL" \
    --revision "$REVISION" \
    --adapter "$ADAPTER" \
    --batch-samples \
    --num-samples 4 \
    --max-new-tokens 512 \
    --temperature "$TEMP_VALUE" \
    --top-p 0.95 \
    --seed 42

  python -m coderl_lab.evaluation \
    --tasks "$DATA" \
    --predictions "$OUT/predictions.jsonl" \
    --output "$OUT/evaluation" \
    --executor docker \
    --timeout 5 \
    --memory 512m \
    --max-workers 4 \
    --k 1 4

  python -m coderl_lab.analysis.evaluation_summary \
    --details "$OUT/evaluation/details.jsonl" \
    --summary "$OUT/evaluation/summary.json" \
    --output "$OUT/evaluation/enhanced_summary.json"
}

run_candidate () {
  local TEMP_TAG="$1"
  local TEMP_VALUE="$2"
  local ARM="$3"
  local ADAPTER="$4"

  run_policy "$TEMP_TAG" "$TEMP_VALUE" "$ARM" "$ADAPTER"

  python -m coderl_lab.analysis.behavior_drift \
    --reference "$ROOT/$TEMP_TAG/sft/predictions.jsonl" \
    --candidate "$ROOT/$TEMP_TAG/$ARM/predictions.jsonl" \
    --output "$ROOT/$TEMP_TAG/$ARM/behavior_vs_sft.json"
}

for SPEC in "t020 0.2" "t050 0.5" "t080 0.8" "t100 1.0"; do
  read -r TAG TEMP <<< "$SPEC"

  run_policy "$TAG" "$TEMP" sft "$SFT"

  run_candidate "$TAG" "$TEMP" scale025_seed101 artifacts/exp004e/perturbations/scale-025/seed-101
  run_candidate "$TAG" "$TEMP" scale050_seed202 artifacts/exp004e/perturbations/scale-050/seed-202
  run_candidate "$TAG" "$TEMP" scale100_seed202 artifacts/exp004d/perturbations/seed-202
  run_candidate "$TAG" "$TEMP" scale200_seed303 artifacts/exp004e/perturbations/scale-200/seed-303
done
