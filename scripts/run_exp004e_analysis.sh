#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-artifacts/exp004e}"
SFT_PRED="${SFT_PRED:-artifacts/exp006b/n16/sft/predictions.jsonl}"
SFT_SUMMARY="${SFT_SUMMARY:-artifacts/exp006b/n16/sft/evaluation/summary.json}"
SFT_ENHANCED="${SFT_ENHANCED:-artifacts/exp006b/n16/sft/evaluation/enhanced_summary.json}"
ANALYSIS="$ROOT/analysis"
mkdir -p "$ANALYSIS"

python -m coderl_lab.analysis.success_concentration \
  --summary "$SFT_SUMMARY" \
  --output "$ANALYSIS/sft_success_concentration.json"

analyze_arm () {
  local arm="$1"
  local eval_dir="$2"
  local out="$ANALYSIS/$arm"
  mkdir -p "$out"

  python -m coderl_lab.analysis.behavior_drift \
    --reference "$SFT_PRED" \
    --candidate "$eval_dir/predictions.jsonl" \
    --output "$out/behavior_vs_sft.json"

  python -m coderl_lab.analysis.success_concentration \
    --summary "$eval_dir/evaluation/summary.json" \
    --output "$out/success_concentration.json"

  python -m coderl_lab.analysis.passk_bootstrap \
    --a "$SFT_SUMMARY" \
    --b "$eval_dir/evaluation/summary.json" \
    --k 1 4 8 16 \
    --iterations 20000 \
    --seed 42 \
    --output "$out/passk_vs_sft.json"
}

for label in 025 050 200; do
  for seed in 101 202 303; do
    arm="scale${label}_seed${seed}"
    analyze_arm "$arm" "$ROOT/eval-validation/$arm"
  done
done

for seed in 101 202 303; do
  arm="scale100_seed${seed}"
  analyze_arm "$arm" "artifacts/exp004d/eval-validation/matched_seed_${seed}"
done

python -m coderl_lab.analysis.perturbation_dose_response \
  --sft-eval "$SFT_ENHANCED" \
  --sft-concentration "$ANALYSIS/sft_success_concentration.json" \
  --analysis-root "$ANALYSIS" \
  --arm "scale025_seed101=$ROOT/eval-validation/scale025_seed101" \
  --arm "scale025_seed202=$ROOT/eval-validation/scale025_seed202" \
  --arm "scale025_seed303=$ROOT/eval-validation/scale025_seed303" \
  --arm "scale050_seed101=$ROOT/eval-validation/scale050_seed101" \
  --arm "scale050_seed202=$ROOT/eval-validation/scale050_seed202" \
  --arm "scale050_seed303=$ROOT/eval-validation/scale050_seed303" \
  --arm "scale100_seed101=artifacts/exp004d/eval-validation/matched_seed_101" \
  --arm "scale100_seed202=artifacts/exp004d/eval-validation/matched_seed_202" \
  --arm "scale100_seed303=artifacts/exp004d/eval-validation/matched_seed_303" \
  --arm "scale200_seed101=$ROOT/eval-validation/scale200_seed101" \
  --arm "scale200_seed202=$ROOT/eval-validation/scale200_seed202" \
  --arm "scale200_seed303=$ROOT/eval-validation/scale200_seed303" \
  --output "$ANALYSIS/summary.json"
