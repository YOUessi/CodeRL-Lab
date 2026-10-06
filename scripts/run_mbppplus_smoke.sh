#!/usr/bin/env bash
set -euo pipefail

TASKS="${TASKS:-data/generated/mbpp-v1/validation_tasks.jsonl}"
PRED="${PRED:-artifacts/exp006b/n16/base/predictions.jsonl}"
OUT="${OUT:-artifacts/exp006c-mbppplus/smoke}"

python -m coderl_lab.analysis.mbppplus_eval \
  --tasks "$TASKS" \
  --predictions "$PRED" \
  --output "$OUT" \
  --max-tasks 2 \
  --max-samples-per-task 2 \
  --workers 2 \
  --version v0.2.0 \
  --k 1 2
