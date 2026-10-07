#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-artifacts/exp004e}"
PERTURB_ROOT="${PERTURB_ROOT:-$ROOT/perturbations}"

for label in 025 050 200; do
  for seed in 101 202 303; do
    ARM="scale${label}_seed${seed}"
    ADAPTER="$PERTURB_ROOT/scale-$label/seed-$seed"
    ROOT="$ROOT/eval-validation" \
    ARM="$ARM" \
    ADAPTER="$ADAPTER" \
    bash scripts/run_exp004c_arm_eval.sh
  done
done
