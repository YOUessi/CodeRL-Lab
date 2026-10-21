#!/usr/bin/env bash
# Build REAL frozen training datasets on an authorized execution node.
# Smoke mode is explicitly non-formal and is never silently used for training.
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE="${1:-sft}"
MODE="${2:-formal}"
source .venv/bin/activate

case "$STAGE" in sft|dpo) ;; *) echo "stage must be sft or dpo" >&2; exit 2;; esac
case "$MODE" in
  smoke)
    # Narrow data-access check; does not claim to have sampled the full source.
    python -m coderl_lab.datasets.posttrain_h4 \
      --stage "$STAGE" \
      --output-dir "data/generated/posttrain-h4-smoke" \
      --train-size 16 --validation-size 8 --smoke-scan-limit 64
    ;;
  formal)
    python -m coderl_lab.datasets.posttrain_h4 \
      --stage "$STAGE" \
      --output-dir "data/generated/posttrain-h4-v1" --seed 42
    ;;
esac
