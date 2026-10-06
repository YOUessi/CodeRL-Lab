#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-artifacts/exp006b/n16}"
EXP006A="${EXP006A:-artifacts/exp006a/eval-validation}"

python -m coderl_lab.analysis.large_k_boundary \
  --base "$ROOT/base/evaluation/summary.json" \
  --sft "$ROOT/sft/evaluation/summary.json" \
  --grpo "$ROOT/grpo/evaluation/summary.json" \
  --output "$ROOT/support_analysis.json"

python -m coderl_lab.analysis.paired_support_bootstrap \
  --a "$ROOT/base/evaluation/summary.json" \
  --b "$ROOT/sft/evaluation/summary.json" \
  --iterations 20000 --seed 42 \
  --output "$ROOT/base_vs_sft_bootstrap.json"

python -m coderl_lab.analysis.paired_support_bootstrap \
  --a "$ROOT/sft/evaluation/summary.json" \
  --b "$ROOT/grpo/evaluation/summary.json" \
  --iterations 20000 --seed 42 \
  --output "$ROOT/sft_vs_grpo_bootstrap.json"

python -m coderl_lab.analysis.track_k4_losses \
  --base-k4 "$EXP006A/base/evaluation/summary.json" \
  --sft-k4 "$EXP006A/sft/evaluation/summary.json" \
  --base-n16 "$ROOT/base/evaluation/summary.json" \
  --sft-n16 "$ROOT/sft/evaluation/summary.json" \
  --grpo-n16 "$ROOT/grpo/evaluation/summary.json" \
  --output "$ROOT/k4_lost_task_tracking.json"
