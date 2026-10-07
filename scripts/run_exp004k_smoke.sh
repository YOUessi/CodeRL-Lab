#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-artifacts/exp004k/smoke}"
TASKS="${TASKS:-data/generated/mbpp-v1/test_tasks.jsonl}"
MODEL="Qwen/Qwen3-1.7B-Base"
REVISION="ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
ADAPTER="${ADAPTER:-artifacts/exp006a/sft-qwen3-1.7b-lora}"

rm -rf "$ROOT"
mkdir -p "$ROOT"

python -m coderl_lab.analysis.heldout_verifier_gated_bottleneck \
  --tasks "$TASKS" \
  --model "$MODEL" \
  --revision "$REVISION" \
  --adapter "$ADAPTER" \
  --output "$ROOT/runner.json" \
  --baseline-predictions "$ROOT/baseline_predictions.jsonl" \
  --public-details "$ROOT/public_only.jsonl" \
  --public-workers 4 \
  --max-tasks 12

python -m coderl_lab.evaluation \
  --tasks "$TASKS" \
  --predictions "$ROOT/baseline_predictions.jsonl" \
  --output "$ROOT/baseline_evaluation" \
  --executor docker \
  --timeout 5 \
  --memory 512m \
  --max-workers 4 \
  --k 1

python -m coderl_lab.analysis.verifier_gated_bottleneck_eval \
  --runner "$ROOT/runner.json" \
  --tasks "$TASKS" \
  --reference-details "$ROOT/baseline_evaluation/details.jsonl" \
  --output "$ROOT/evaluation" \
  --workers 4 \
  --timeout 5 \
  --memory 512m \
  --iterations 2000 \
  --seed 42
