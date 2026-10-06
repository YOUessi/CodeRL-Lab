#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/generated/mbpp-v1}"
ROOT="${ROOT:-artifacts/exp006a/base-smoke}"
MODEL="Qwen/Qwen3-1.7B-Base"
REVISION="ea980cb0a6c2ae4b936e82123acc929f1cec04c1"

if [ ! -f "$DATA_DIR/validation_tasks.jsonl" ]; then
  bash scripts/prepare_mbpp.sh
fi

mkdir -p "$ROOT"

python -m coderl_lab.generation \
  --tasks "$DATA_DIR/validation_tasks.jsonl" \
  --output "$ROOT/predictions.jsonl" \
  --metadata-output "$ROOT/generation.json" \
  --model "$MODEL" \
  --revision "$REVISION" \
  --max-tasks 20 \
  --batch-samples \
  --num-samples 4 \
  --max-new-tokens 512 \
  --temperature 0.8 \
  --top-p 0.95 \
  --seed 42

python -m coderl_lab.evaluation \
  --tasks "$DATA_DIR/validation_tasks.jsonl" \
  --predictions "$ROOT/predictions.jsonl" \
  --output "$ROOT/evaluation" \
  --executor docker \
  --timeout 5 \
  --memory 512m \
  --max-workers 4 \
  --k 1 4

python -m coderl_lab.analysis.evaluation_summary \
  --details "$ROOT/evaluation/details.jsonl" \
  --summary "$ROOT/evaluation/summary.json" \
  --output "$ROOT/evaluation/enhanced_summary.json"
