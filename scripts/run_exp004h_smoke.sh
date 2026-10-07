#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-artifacts/exp004h/smoke}"
DATA="${DATA:-data/generated/mbpp-v1/validation_tasks.jsonl}"
MODEL="Qwen/Qwen3-1.7B-Base"
REVISION="ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
SFT="${SFT:-artifacts/exp006a/sft-qwen3-1.7b-lora}"

rm -rf "$ROOT"
mkdir -p "$ROOT"

python -m coderl_lab.analysis.local_bottleneck_intervention \
  --tasks "$DATA" \
  --model "$MODEL" \
  --revision "$REVISION" \
  --reference-adapter "$SFT" \
  --candidate-adapter artifacts/exp004e/perturbations/scale-025/seed-101 \
  --reference-predictions artifacts/exp004f/greedy/sft.jsonl \
  --baseline-predictions artifacts/exp004f/greedy/scale025_seed101/predictions.jsonl \
  --output "$ROOT/scale025_seed101.json" \
  --max-tasks 4 \
  --primary-window 128 \
  --low-threshold 0.05 \
  --high-threshold 0.20 \
  --bias 0.25
