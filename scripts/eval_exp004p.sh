#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv-gpu/bin/python}"
ROOT="${ROOT:-artifacts/exp004p}"
LCB="${LCB_ROOT:-.external/livecodebench}"
PYTHONPATH=src "$PYTHON" -m coderl_lab.analysis.exp004p_formal_eval \
 --runner "$ROOT/sft512/runner.json" \
 --public-tasks "$ROOT/data/public_tasks.jsonl" \
 --manifest "$ROOT/data/manifest.json" \
 --lcb-repo "$LCB" --output-dir "$ROOT/final_evaluation" \
 --expected-tasks "${EXPECTED_TASKS:-101}" --iterations 20000 --seed 42
