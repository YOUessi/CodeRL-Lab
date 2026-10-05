#!/usr/bin/env bash
set -euo pipefail

python -m coderl_lab.evaluation \
  --tasks benchmarks/sample_tasks.jsonl \
  --predictions benchmarks/sample_predictions.jsonl \
  --output results/exp001-smoke \
  --executor local \
  --allow-unsafe-local \
  --k 1 2
