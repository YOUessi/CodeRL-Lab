#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-artifacts/exp004d}"
SFT_PRED="${SFT_PRED:-artifacts/exp006b/n16/sft/predictions.jsonl}"
SFT_SUMMARY="${SFT_SUMMARY:-artifacts/exp006b/n16/sft/evaluation/summary.json}"
ANALYSIS="$ROOT/analysis"
mkdir -p "$ANALYSIS"

python -m coderl_lab.analysis.success_concentration   --summary "$SFT_SUMMARY"   --output "$ANALYSIS/sft_success_concentration.json"

for SEED in 101 202 303; do
  ARM="matched_seed_$SEED"
  DIR="$ROOT/eval-validation/$ARM"
  OUT="$ANALYSIS/$ARM"
  mkdir -p "$OUT"

  python -m coderl_lab.analysis.behavior_drift     --reference "$SFT_PRED"     --candidate "$DIR/predictions.jsonl"     --output "$OUT/behavior_vs_sft.json"

  python -m coderl_lab.analysis.success_concentration     --summary "$DIR/evaluation/summary.json"     --output "$OUT/success_concentration.json"

  python -m coderl_lab.analysis.passk_bootstrap     --a "$SFT_SUMMARY"     --b "$DIR/evaluation/summary.json"     --k 1 4 8 16     --iterations 20000     --seed 42     --output "$OUT/passk_vs_sft.json"
done
