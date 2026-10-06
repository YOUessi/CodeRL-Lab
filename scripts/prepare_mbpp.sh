#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-data/generated/mbpp-v1}"
REVISION="${REVISION:-4bb6404fdc6cacfda99d4ac4205087b89d32030c}"

python -m coderl_lab.datasets.mbpp \
  --output-dir "$OUTPUT_DIR" \
  --revision "$REVISION" \
  --train-public-tests 2 \
  --eval-public-tests 1
