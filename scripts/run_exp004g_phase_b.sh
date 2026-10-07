#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-artifacts/exp004g}"
TASKS="${TASKS:-data/generated/mbpp-v1/validation_tasks.jsonl}"
MODEL="Qwen/Qwen3-1.7B-Base"
REVISION="ea980cb0a6c2ae4b936e82123acc929f1cec04c1"

test -f "$ROOT/sft_margin_profile.json"
test -f "$ROOT/susceptibility_labels.json"

python -m coderl_lab.analysis.trajectory_margin_profile \
  --tasks "$TASKS" \
  --model "$MODEL" \
  --revision "$REVISION" \
  --output "$ROOT/base_margin_profile.json" \
  --primary-window 128 \
  --max-new-tokens 512

python -m coderl_lab.analysis.margin_shift_statistics \
  --base-profile "$ROOT/base_margin_profile.json" \
  --sft-profile "$ROOT/sft_margin_profile.json" \
  --susceptibility "$ROOT/susceptibility_labels.json" \
  --output "$ROOT/phase_b_analysis.json" \
  --iterations 20000 \
  --seed 42
