#!/usr/bin/env bash
set -euo pipefail

OUT="${OUT:-artifacts/exp005b/comparison}"
mkdir -p "$OUT"

python -m coderl_lab.analysis.exp005b_compare \
  --exp005a-summary results/exp005a/summary.json \
  --online-training artifacts/exp005b/grpo-online-qwen3-0.6b/run_summary.json \
  --online-dynamics artifacts/exp005b/grpo-online-qwen3-0.6b/dynamics_summary.json \
  --online-evaluation artifacts/exp005b/eval-validation/online_dynamic/evaluation/enhanced_summary.json \
  --output "$OUT/summary.json"

for BASELINE in full_random random_subset static_boundary; do
  case "$BASELINE" in
    full_random)
      BASE="artifacts/exp003/eval-validation/grpo/evaluation/summary.json"
      ;;
    random_subset)
      BASE="artifacts/exp005/eval-validation/random_control/evaluation/summary.json"
      ;;
    static_boundary)
      BASE="artifacts/exp005/eval-validation/boundary/evaluation/summary.json"
      ;;
  esac

  python -m coderl_lab.analysis.paired_bootstrap \
    --a "$BASE" \
    --b artifacts/exp005b/eval-validation/online_dynamic/evaluation/summary.json \
    --iterations 20000 \
    --seed 42 \
    --output "$OUT/bootstrap_online_minus_${BASELINE}.json"
done
