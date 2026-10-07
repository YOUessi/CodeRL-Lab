#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-artifacts/exp004b/import-pairs}"
TASKS="${TASKS:-data/generated/mbpp-v1/train_tasks.jsonl}"
PREDICTIONS="${PREDICTIONS:-artifacts/exp004a/process-screen-sft-1.7b/predictions.jsonl}"
DIAGNOSTICS="${DIAGNOSTICS:-artifacts/exp004a/process-screen-sft-1.7b/scoring/details.jsonl}"

mkdir -p "$ROOT"

python -m coderl_lab.data.runtime_repair_preferences \
  --tasks "$TASKS" \
  --predictions "$PREDICTIONS" \
  --diagnostics "$DIAGNOSTICS" \
  --output "$ROOT/preferences.jsonl" \
  --summary "$ROOT/summary.json" \
  --timeout 5 \
  --memory 512m
