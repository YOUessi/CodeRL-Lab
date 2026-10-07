#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-artifacts/exp004f}"
DATA="${DATA:-data/generated/mbpp-v1/validation_tasks.jsonl}"
MODEL="Qwen/Qwen3-1.7B-Base"
REVISION="ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
SFT="${SFT:-artifacts/exp006a/sft-qwen3-1.7b-lora}"

mkdir -p "$ROOT/divergence"

run_arm () {
  ARM="$1"
  ADAPTER="$2"

  python -m coderl_lab.analysis.greedy_first_divergence     --tasks "$DATA"     --model "$MODEL"     --revision "$REVISION"     --reference-adapter "$SFT"     --candidate-adapter "$ADAPTER"     --reference-predictions "$ROOT/greedy/sft.jsonl"     --candidate-predictions "$ROOT/greedy/$ARM/predictions.jsonl"     --output "$ROOT/divergence/$ARM.json"
}

run_arm scale025_seed101 artifacts/exp004e/perturbations/scale-025/seed-101
run_arm scale050_seed202 artifacts/exp004e/perturbations/scale-050/seed-202
run_arm scale100_seed202 artifacts/exp004d/perturbations/seed-202
run_arm scale200_seed303 artifacts/exp004e/perturbations/scale-200/seed-303
