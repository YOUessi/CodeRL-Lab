#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-artifacts/exp004b/random-label-control}"
SOURCE="${SOURCE:-artifacts/exp004b/import-pairs/preferences_one_per_task.jsonl}"
SEED="${SEED:-42}"

mkdir -p "$ROOT"

python -m coderl_lab.data.randomize_preferences \
  --input "$SOURCE" \
  --output "$ROOT/preferences.jsonl" \
  --summary "$ROOT/summary.json" \
  --seed "$SEED" \
  --swap-fraction 0.5
