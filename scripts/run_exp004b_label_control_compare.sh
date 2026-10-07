#!/usr/bin/env bash
set -euo pipefail

SFT="${SFT:-artifacts/exp006b/n16/sft/evaluation/summary.json}"
SEMANTIC="${SEMANTIC:-artifacts/exp004b/eval-validation/dpo/evaluation/summary.json}"
RANDOM="${RANDOM:-artifacts/exp004b/eval-validation/random_control/evaluation/summary.json}"
OUT="${OUT:-artifacts/exp004b/label-control-compare}"

mkdir -p "$OUT"

python -m coderl_lab.analysis.passk_bootstrap \
  --a "$SFT" --b "$SEMANTIC" \
  --k 1 4 8 16 --iterations 20000 --seed 42 \
  --output "$OUT/semantic_vs_sft.json"

python -m coderl_lab.analysis.passk_bootstrap \
  --a "$SFT" --b "$RANDOM" \
  --k 1 4 8 16 --iterations 20000 --seed 42 \
  --output "$OUT/random_vs_sft.json"

python -m coderl_lab.analysis.passk_bootstrap \
  --a "$RANDOM" --b "$SEMANTIC" \
  --k 1 4 8 16 --iterations 20000 --seed 42 \
  --output "$OUT/semantic_vs_random.json"
