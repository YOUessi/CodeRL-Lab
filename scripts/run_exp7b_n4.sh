#!/usr/bin/env bash
# 7B historical v4 exploratory study: generate all 404 candidate codes before private scoring.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv-gpu/bin/python}"
ROOT="${ROOT:-artifacts/exp7b/n4}"
DATA="${PUBLIC_TASKS:-../CodeRL-Lab-track-exp004p/artifacts/exp004p/data/public_tasks.jsonl}"
LCB="${LCB_ROOT:-.external/livecodebench}"
test -x "$PYTHON"
test "$(git -C "$LCB" rev-parse HEAD)" = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
PYTHONPATH=src "$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 11000
PYTHONPATH=src "$PYTHON" -m coderl_lab.analysis.exp7b_public_n4 \
  --public-tasks "$DATA" --lcb-repo "$LCB" --output-dir "$ROOT/generation" --samples-per-task 4
# Official private tests enter the process only after all 404 original codes,
# their exact hashes, and the generation identity have been frozen.
PYTHONPATH=src "$PYTHON" -m coderl_lab.analysis.exp7b_official_n4 \
  --generation-dir "$ROOT/generation" --public-tasks "$DATA" \
  --lcb-repo "$LCB" --output-dir "$ROOT/official"
