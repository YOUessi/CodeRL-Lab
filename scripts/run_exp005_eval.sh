#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/generated/mbpp-v1}"
ROOT="${ROOT:-artifacts/exp005/eval-validation}"
MODEL="Qwen/Qwen3-0.6B-Base"
REVISION="da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
ADAPTER="${ADAPTER:-artifacts/exp005/grpo-boundary-qwen3-0.6b}"

if [ ! -f "$ADAPTER/adapter_model.safetensors" ]; then
  echo "Missing EXP-005 adapter: $ADAPTER" >&2
  exit 2
fi

mkdir -p "$ROOT/boundary_grpo"

python -m coderl_lab.generation \
  --tasks "$DATA_DIR/validation_tasks.jsonl" \
  --output "$ROOT/boundary_grpo/predictions.jsonl" \
  --metadata-output "$ROOT/boundary_grpo/generation.json" \
  --model "$MODEL" \
  --revision "$REVISION" \
  --adapter "$ADAPTER" \
  --batch-samples \
  --num-samples 4 \
  --max-new-tokens 512 \
  --temperature 0.8 \
  --top-p 0.95 \
  --seed 42

python -m coderl_lab.evaluation \
  --tasks "$DATA_DIR/validation_tasks.jsonl" \
  --predictions "$ROOT/boundary_grpo/predictions.jsonl" \
  --output "$ROOT/boundary_grpo/evaluation" \
  --executor docker \
  --timeout 5 \
  --memory 512m \
  --k 1 4

python -m coderl_lab.analysis.evaluation_summary \
  --details "$ROOT/boundary_grpo/evaluation/details.jsonl" \
  --summary "$ROOT/boundary_grpo/evaluation/summary.json" \
  --output "$ROOT/boundary_grpo/enhanced_summary.json"
