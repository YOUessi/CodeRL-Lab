#!/usr/bin/env bash
set -euo pipefail

python -m coderl_lab.evaluation \
  --tasks benchmarks/sample_tasks_v2.jsonl \
  --predictions benchmarks/sample_predictions_v2.jsonl \
  --output results/data001-dual-mode-smoke \
  --executor local \
  --allow-unsafe-local \
  --k 1 2
