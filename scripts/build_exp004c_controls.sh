#!/usr/bin/env bash
set -euo pipefail

SOURCE="${SOURCE:-artifacts/exp004b/import-pairs/preferences_one_per_task.jsonl}"
ROOT="${ROOT:-artifacts/exp004c/controls}"

test -f "$SOURCE"
mkdir -p "$ROOT/reverse" "$ROOT/noop"

python -m coderl_lab.data.preference_controls \
  --input "$SOURCE" \
  --output "$ROOT/reverse/preferences.jsonl" \
  --summary "$ROOT/reverse/summary.json" \
  --mode reverse

python -m coderl_lab.data.preference_controls \
  --input "$SOURCE" \
  --output "$ROOT/noop/preferences.jsonl" \
  --summary "$ROOT/noop/summary.json" \
  --mode noop \
  --noop-side chosen
