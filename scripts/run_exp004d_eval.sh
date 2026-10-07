#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-artifacts/exp004d}"
PERTURB_ROOT="${PERTURB_ROOT:-$ROOT/perturbations}"

for SEED in 101 202 303; do
  ARM="matched_seed_${SEED}"
  ADAPTER="$PERTURB_ROOT/seed-$SEED"
  ROOT="$ROOT/eval-validation" \
  ARM="$ARM" \
  ADAPTER="$ADAPTER" \
  bash scripts/run_exp004c_arm_eval.sh
done
