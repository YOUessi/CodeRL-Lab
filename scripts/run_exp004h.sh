#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-artifacts/exp004h}"
DATA="${DATA:-data/generated/mbpp-v1/validation_tasks.jsonl}"
MODEL="Qwen/Qwen3-1.7B-Base"
REVISION="ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
SFT="${SFT:-artifacts/exp006a/sft-qwen3-1.7b-lora}"

rm -rf "$ROOT/arms" "$ROOT/analysis"
mkdir -p "$ROOT/arms" "$ROOT/analysis"

run_arm () {
  NAME="$1"
  ADAPTER="$2"
  BASELINE="$3"

  python -m coderl_lab.analysis.local_bottleneck_intervention \
    --tasks "$DATA" \
    --model "$MODEL" \
    --revision "$REVISION" \
    --reference-adapter "$SFT" \
    --candidate-adapter "$ADAPTER" \
    --reference-predictions artifacts/exp004f/greedy/sft.jsonl \
    --baseline-predictions "$BASELINE" \
    --output "$ROOT/arms/$NAME.json" \
    --primary-window 128 \
    --low-threshold 0.05 \
    --high-threshold 0.20 \
    --bias 0.25
}

run_arm scale025_seed101 \
  artifacts/exp004e/perturbations/scale-025/seed-101 \
  artifacts/exp004f/greedy/scale025_seed101/predictions.jsonl

run_arm scale050_seed202 \
  artifacts/exp004e/perturbations/scale-050/seed-202 \
  artifacts/exp004f/greedy/scale050_seed202/predictions.jsonl

run_arm scale100_seed202 \
  artifacts/exp004d/perturbations/seed-202 \
  artifacts/exp004f/greedy/scale100_seed202/predictions.jsonl

run_arm scale200_seed303 \
  artifacts/exp004e/perturbations/scale-200/seed-303 \
  artifacts/exp004f/greedy/scale200_seed303/predictions.jsonl

python -m coderl_lab.analysis.local_bottleneck_statistics \
  --arm "$ROOT/arms/scale025_seed101.json" \
  --arm "$ROOT/arms/scale050_seed202.json" \
  --arm "$ROOT/arms/scale100_seed202.json" \
  --arm "$ROOT/arms/scale200_seed303.json" \
  --output "$ROOT/analysis/summary.json" \
  --iterations 20000 \
  --seed 42
