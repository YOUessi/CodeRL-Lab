#!/usr/bin/env bash
set -euo pipefail

OUTPUT="${OUTPUT:-artifacts/data001/apps-sample.jsonl}"
REPORT="${REPORT:-results/data001-apps-sample/report.json}"
LIMIT="${LIMIT:-50}"

python -m coderl_lab.dataset.apps \
  --split train \
  --output "$OUTPUT" \
  --report "$REPORT" \
  --limit "$LIMIT" \
  --seed 42 \
  --min-tests 4
