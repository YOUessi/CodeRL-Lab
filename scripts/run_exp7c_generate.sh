#!/usr/bin/env bash
# EXP-7C frozen 164 x 4 two-arm HumanEval+ generator (Tang GPU).
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv-gpu/bin/python}"
DATA="${DATA:-artifacts/exp7c/data}"
ADAPTER="${ADAPTER:-artifacts/train011/qlora-ultrachat}"
OUTPUT="${OUTPUT:-artifacts/exp7c/formal}"
test -x "$PYTHON" || { echo "Missing Python: $PYTHON" >&2; exit 2; }
test -s "$DATA/manifest.json"
test -s "$DATA/public_tasks.jsonl"
test -s "$ADAPTER/adapter_model.safetensors"
if [ -s "$OUTPUT/summary.json" ]; then
    echo "Formal generation already completed; refusing overwrite" >&2
    exit 5
fi
(
    flock -n 9 || { echo "Another GPU experiment holds the lock" >&2; exit 9; }
    PYTHONPATH=src "$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 11000
    PYTHONPATH=src "$PYTHON" -m coderl_lab.analysis.exp7c_matched_generate \
        --public-tasks "$DATA/public_tasks.jsonl" \
        --manifest "$DATA/manifest.json" \
        --adapter "$ADAPTER" \
        --output-dir "$OUTPUT" \
        --samples-per-task 4
) 9>/tmp/coderl_lab_gpu_experiment_lock
echo "EXP-7C 1312 paired frozen samples generated"
