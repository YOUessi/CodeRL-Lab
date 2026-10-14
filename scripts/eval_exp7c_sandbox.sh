#!/usr/bin/env bash
# EXP-7C matched 7B Base/SFT HumanEval+ isolated assertions after generation freeze.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv-gpu/bin/python}"
DATA="${DATA:-artifacts/exp7c/data}"
SOURCE="${SOURCE:-artifacts/exp7c/source/test.jsonl}"
GENERATED="${GENERATED:-artifacts/exp7c/formal}"
OUTPUT="${OUTPUT:-artifacts/exp7c/evaluation}"
test -x "$PYTHON"
test -s "$GENERATED/summary.json" || { echo "All 1312 candidates must be frozen first" >&2; exit 3; }
test -s "$GENERATED/checkpoint.jsonl"
test -s "$SOURCE"
PYTHONPATH=src "$PYTHON" -m coderl_lab.analysis.exp7c_sandbox_score \
    --generation-dir "$GENERATED" \
    --public-tasks "$DATA/public_tasks.jsonl" \
    --manifest "$DATA/manifest.json" \
    --source-jsonl "$SOURCE" \
    --output-dir "$OUTPUT"
