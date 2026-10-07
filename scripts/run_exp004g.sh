#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-artifacts/exp004g}"
TASKS="${TASKS:-data/generated/mbpp-v1/validation_tasks.jsonl}"
GREEDY_ROOT="${GREEDY_ROOT:-artifacts/exp004f/greedy}"
MODEL="Qwen/Qwen3-1.7B-Base"
REVISION="ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
SFT="${SFT:-artifacts/exp006a/sft-qwen3-1.7b-lora}"

mkdir -p "$ROOT"

python -m coderl_lab.analysis.trajectory_margin_profile \
  --tasks "$TASKS" \
  --model "$MODEL" \
  --revision "$REVISION" \
  --adapter "$SFT" \
  --greedy-predictions "$GREEDY_ROOT/sft.jsonl" \
  --output "$ROOT/sft_margin_profile.json" \
  --primary-window 128 \
  --max-new-tokens 512

python -m coderl_lab.analysis.trajectory_susceptibility \
  --greedy-root "$GREEDY_ROOT" \
  --output "$ROOT/susceptibility_labels.json"

python -m coderl_lab.analysis.susceptibility_statistics \
  --margin-profile "$ROOT/sft_margin_profile.json" \
  --susceptibility "$ROOT/susceptibility_labels.json" \
  --output "$ROOT/analysis.json" \
  --iterations 20000 \
  --seed 42
