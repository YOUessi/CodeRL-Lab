#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
ROOT="${ROOT:-artifacts/exp004n}"
PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" \
  -m coderl_lab.analysis.exp004n_formal_eval \
  --sft512 "$ROOT/sft512/runner.json" \
  --base512 "$ROOT/base512/runner.json" \
  --sft1024 "$ROOT/sft1024/runner.json" \
  --public-tasks "$ROOT/data/public_tasks.jsonl" \
  --manifest "$ROOT/data/manifest.json" \
  --adapter "${ADAPTER:-artifacts/exp006a/sft-qwen3-1.7b-lora}" \
  --lcb-repo "${LCB_ROOT:-.external/livecodebench}" \
  --output-dir "$ROOT/final_evaluation" \
  --iterations 20000 --seed 42
