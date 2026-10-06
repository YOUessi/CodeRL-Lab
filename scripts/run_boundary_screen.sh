#!/usr/bin/env bash
set -euo pipefail

TASKS="${TASKS:-data/generated/mbpp-v1/train_tasks.jsonl}"
ROOT="${ROOT:-artifacts/exp005/screen-v1}"
MODEL="Qwen/Qwen3-0.6B-Base"
REVISION="da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
SFT_ADAPTER="${SFT_ADAPTER:-artifacts/exp002/sft-qwen3-0.6b-lora}"
EXPECTED_SFT_SHA="7fcb2b0ca7608be980fb4cbae5158241886ac67f3cc2f5b855cf34531c5982f7"

if [ ! -f "$TASKS" ]; then
  bash scripts/prepare_mbpp.sh
fi

if [ ! -f "$SFT_ADAPTER/adapter_model.safetensors" ]; then
  echo "Missing EXP-002 SFT adapter: $SFT_ADAPTER" >&2
  exit 2
fi

ACTUAL_SFT_SHA="$(sha256sum "$SFT_ADAPTER/adapter_model.safetensors" | awk '{print $1}')"
if [ "$ACTUAL_SFT_SHA" != "$EXPECTED_SFT_SHA" ]; then
  echo "SFT adapter SHA mismatch" >&2
  echo "expected: $EXPECTED_SFT_SHA" >&2
  echo "actual:   $ACTUAL_SFT_SHA" >&2
  exit 3
fi

docker image inspect python:3.11-slim >/dev/null 2>&1 || docker pull python:3.11-slim
mkdir -p "$ROOT"

python -m coderl_lab.generation \
  --tasks "$TASKS" \
  --output "$ROOT/predictions.jsonl" \
  --metadata-output "$ROOT/generation.json" \
  --model "$MODEL" \
  --revision "$REVISION" \
  --adapter "$SFT_ADAPTER" \
  --batch-samples \
  --num-samples 4 \
  --max-new-tokens 256 \
  --temperature 0.8 \
  --top-p 0.95 \
  --seed 42

python -m coderl_lab.sampling.boundary \
  --tasks "$TASKS" \
  --predictions "$ROOT/predictions.jsonl" \
  --output-dir "$ROOT" \
  --num-samples 4 \
  --timeout 5 \
  --memory 512m \
  --max-workers 4
