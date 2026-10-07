#!/usr/bin/env bash
set -euo pipefail

SOURCE="${SOURCE:-artifacts/exp004b/import-pairs/preferences_one_per_task.jsonl}"
ROOT="${ROOT:-artifacts/exp004b/random-label-control}"

mkdir -p "$ROOT"

python -m coderl_lab.data.randomize_preferences \
  --input "$SOURCE" \
  --output "$ROOT/preferences.jsonl" \
  --summary "$ROOT/summary.json" \
  --seed 42042 \
  --swap-fraction 0.5
