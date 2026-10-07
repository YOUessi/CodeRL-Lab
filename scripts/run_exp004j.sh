#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-artifacts/exp004j/full}"
TASKS="${TASKS:-data/generated/mbpp-v1/validation_tasks.jsonl}"
MODEL="Qwen/Qwen3-1.7B-Base"
REVISION="ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
ADAPTER="${ADAPTER:-artifacts/exp006a/sft-qwen3-1.7b-lora}"
REF_PRED="${REF_PRED:-artifacts/exp004f/greedy/sft.jsonl}"
REF_DETAILS="${REF_DETAILS:-artifacts/exp004f/greedy/sft_eval/details.jsonl}"

rm -rf "$ROOT"
mkdir -p "$ROOT"

python -m coderl_lab.analysis.verifier_gated_bottleneck \
  --tasks "$TASKS" \
  --model "$MODEL" \
  --revision "$REVISION" \
  --adapter "$ADAPTER" \
  --reference-predictions "$REF_PRED" \
  --reference-details "$REF_DETAILS" \
  --output "$ROOT/runner.json"

python -m coderl_lab.analysis.verifier_gated_bottleneck_eval \
  --runner "$ROOT/runner.json" \
  --tasks "$TASKS" \
  --reference-details "$REF_DETAILS" \
  --output "$ROOT/evaluation" \
  --workers 4 \
  --iterations 20000 \
  --seed 42
