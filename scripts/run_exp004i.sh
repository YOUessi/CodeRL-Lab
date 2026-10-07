#!/usr/bin/env bash
set -euo pipefail

TASKS="${TASKS:-data/generated/mbpp-v1/validation_tasks.jsonl}"
ROOT="${ROOT:-artifacts/exp004i}"
H_ROOT="${H_ROOT:-artifacts/exp004h/arms}"
F_ROOT="${F_ROOT:-artifacts/exp004f/greedy}"

rm -rf "$ROOT"
mkdir -p "$ROOT"

python -m coderl_lab.analysis.bottleneck_correctness_causality \
  --tasks "$TASKS" \
  --reference-predictions "$F_ROOT/sft.jsonl" \
  --reference-details "$F_ROOT/sft_eval/details.jsonl" \
  --greedy-root "$F_ROOT" \
  --arm "$H_ROOT/scale025_seed101.json" \
  --arm "$H_ROOT/scale050_seed202.json" \
  --arm "$H_ROOT/scale100_seed202.json" \
  --arm "$H_ROOT/scale200_seed303.json" \
  --output "$ROOT" \
  --workers 4 \
  --timeout 5 \
  --memory 512m \
  --iterations 20000 \
  --seed 42
