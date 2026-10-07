#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-artifacts/exp004f/temperature}"
GREEDY="${GREEDY:-artifacts/exp004f/summary.json}"

python -m coderl_lab.analysis.temperature_sweep_summary \
  --root "$ROOT" \
  --greedy-summary "$GREEDY" \
  --iterations 20000 \
  --seed 42 \
  --output "$ROOT/summary.json"
