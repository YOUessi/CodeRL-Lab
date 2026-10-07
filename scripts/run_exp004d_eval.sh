#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-artifacts/exp004d}"
EVAL_ROOT="${EVAL_ROOT:-artifacts/exp004d/eval-validation}"

for SEED in 101 202 303; do
  ARM="random_$SEED"   ADAPTER="$ROOT/random-$SEED"   ROOT="$EVAL_ROOT"   bash scripts/run_exp004c_arm_eval.sh
done
