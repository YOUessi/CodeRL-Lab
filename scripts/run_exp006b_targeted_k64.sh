#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/generated/mbpp-v1}"
N16_ROOT="${N16_ROOT:-artifacts/exp006b/n16}"
ROOT="${ROOT:-artifacts/exp006b/k64-targeted}"
MODEL="Qwen/Qwen3-1.7B-Base"
REVISION="ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
SFT="${SFT:-artifacts/exp006a/sft-qwen3-1.7b-lora}"
GRPO="${GRPO:-artifacts/exp006a/grpo-qwen3-1.7b}"

SUPPORT="$N16_ROOT/support_analysis.json"
TARGETS="$ROOT/target_tasks.jsonl"
SEEDS="$ROOT/task_seed_map.json"

if [ ! -f "$SUPPORT" ]; then
  echo "Missing n=16 support analysis: $SUPPORT" >&2
  exit 2
fi

mkdir -p "$ROOT"

python -m coderl_lab.analysis.large_k_extend \
  --support-analysis "$SUPPORT" \
  --tasks "$DATA_DIR/validation_tasks.jsonl" \
  --base-predictions "$N16_ROOT/base/predictions.jsonl" \
  --output-tasks "$TARGETS" \
  --output-seeds "$SEEDS"

TARGET_COUNT="$(grep -cve '^$' "$TARGETS" || true)"
if [ "$TARGET_COUNT" -eq 0 ]; then
  echo "No Base-solved/SFT-unsolved targets remain at n=16."
  exit 0
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
    --tasks "$TARGETS" \
    --output "$OUT/predictions.jsonl" \
    --metadata-output "$OUT/generation.json" \
    --model "$MODEL" \
    --revision "$REVISION" \
    "${EXTRA[@]}" \
    --batch-samples \
    --sample-batch-size 16 \
    --task-seed-map "$SEEDS" \
    --num-samples 64 \
    --max-new-tokens 512 \
    --temperature 0.8 \
    --top-p 0.95 \
    --seed 42

  python -m coderl_lab.analysis.verify_large_k_prefix \
    --original "$N16_ROOT/$MODE/predictions.jsonl" \
    --extended "$OUT/predictions.jsonl" \
    --targets "$TARGETS" \
    --prefix-samples 16 \
    --output "$OUT/prefix_verification.json"

  python -m coderl_lab.evaluation \
    --tasks "$TARGETS" \
    --predictions "$OUT/predictions.jsonl" \
    --output "$OUT/evaluation" \
    --executor docker \
    --timeout 5 \
    --memory 512m \
    --max-workers 4 \
    --k 1 4 8 16 32 64

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
