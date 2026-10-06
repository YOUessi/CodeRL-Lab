#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/generated/mbpp-v1}"
ROOT="${ROOT:-artifacts/exp006b/n16}"
MODEL="Qwen/Qwen3-1.7B-Base"
REVISION="ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
SFT="${SFT:-artifacts/exp006a/sft-qwen3-1.7b-lora}"
GRPO="${GRPO:-artifacts/exp006a/grpo-qwen3-1.7b}"

if [ ! -f "$SFT/adapter_model.safetensors" ]; then
  echo "Missing SFT adapter: $SFT" >&2
  exit 2
fi
if [ ! -f "$GRPO/adapter_model.safetensors" ]; then
  echo "Missing GRPO adapter: $GRPO" >&2
  exit 2
fi

for MODE in base sft grpo; do
  OUT="$ROOT/$MODE"
  rm -rf "$OUT"
  mkdir -p "$OUT"

  EXTRA=()
  if [ "$MODE" = "sft" ]; then
    EXTRA=(--adapter "$SFT")
  elif [ "$MODE" = "grpo" ]; then
    EXTRA=(--adapter "$GRPO")
  fi

  python -m coderl_lab.generation \
    --tasks "$DATA_DIR/validation_tasks.jsonl" \
    --output "$OUT/predictions.jsonl" \
    --metadata-output "$OUT/generation.json" \
    --model "$MODEL" \
    --revision "$REVISION" \
    "${EXTRA[@]}" \
    --batch-samples \
    --num-samples 16 \
    --max-new-tokens 512 \
    --temperature 0.8 \
    --top-p 0.95 \
    --seed 42

  python -m coderl_lab.evaluation \
    --tasks "$DATA_DIR/validation_tasks.jsonl" \
    --predictions "$OUT/predictions.jsonl" \
    --output "$OUT/evaluation" \
    --executor docker \
    --timeout 5 \
    --memory 512m \
    --max-workers 4 \
    --k 1 4 8 16

  python -m coderl_lab.analysis.evaluation_summary \
    --details "$OUT/evaluation/details.jsonl" \
    --summary "$OUT/evaluation/summary.json" \
    --output "$OUT/evaluation/enhanced_summary.json"
done

python -m coderl_lab.analysis.large_k_boundary \
  --base "$ROOT/base/evaluation/summary.json" \
  --sft "$ROOT/sft/evaluation/summary.json" \
  --grpo "$ROOT/grpo/evaluation/summary.json" \
  --output "$ROOT/support_analysis.json"
