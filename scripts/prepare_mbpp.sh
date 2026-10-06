#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-data/generated/mbpp-v1}"
REVISION="${REVISION:-main}"

python -m coderl_lab.datasets.mbpp \
  --output-dir "$OUTPUT_DIR" \
  --revision "$REVISION" \
  --train-public-tests 2 \
  --eval-public-tests 1
