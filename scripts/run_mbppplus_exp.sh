#!/usr/bin/env bash
set -euo pipefail

TASKS="${TASKS:-data/generated/mbpp-v1/validation_tasks.jsonl}"
ROOT="${ROOT:-artifacts/exp006c-mbppplus}"

for MODE in base sft grpo; do
  PRED="artifacts/exp006b/n16/$MODE/predictions.jsonl"
  OUT="$ROOT/$MODE"
  if [ ! -f "$PRED" ]; then
    echo "Missing fixed n=16 predictions: $PRED" >&2
    exit 2
  fi

  python -m coderl_lab.analysis.mbppplus_eval \
    --tasks "$TASKS" \
    --predictions "$PRED" \
    --output "$OUT" \
    --workers 8 \
    --version v0.2.0 \
    --k 1 4 8 16
done
