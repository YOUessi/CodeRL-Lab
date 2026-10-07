#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-artifacts/exp004f}"
DATA="${DATA:-data/generated/mbpp-v1/validation_tasks.jsonl}"
MODEL="Qwen/Qwen3-1.7B-Base"
REVISION="ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
SFT="${SFT:-artifacts/exp006a/sft-qwen3-1.7b-lora}"
CACHE="$ROOT/reference_prompt_logits.pt"

adapter_for () {
  scale="$1"
  seed="$2"
  if [ "$scale" = "100" ]; then
    echo "artifacts/exp004d/perturbations/seed-$seed"
  else
    echo "artifacts/exp004e/perturbations/scale-$scale/seed-$seed"
  fi
}

rm -rf "$ROOT/greedy" "$ROOT/logits"
mkdir -p "$ROOT/greedy" "$ROOT/logits"

python -m coderl_lab.generation   --tasks "$DATA"   --output "$ROOT/greedy/sft.jsonl"   --metadata-output "$ROOT/greedy/sft_generation.json"   --model "$MODEL"   --revision "$REVISION"   --adapter "$SFT"   --num-samples 1   --max-new-tokens 512   --greedy

python -m coderl_lab.evaluation   --tasks "$DATA"   --predictions "$ROOT/greedy/sft.jsonl"   --output "$ROOT/greedy/sft_eval"   --executor docker   --timeout 5   --memory 512m   --max-workers 4   --k 1

for SCALE in 025 050 100 200; do
  for SEED in 101 202 303; do
    ARM="scale${SCALE}_seed${SEED}"
    ADAPTER="$(adapter_for "$SCALE" "$SEED")"
    DIR="$ROOT/greedy/$ARM"
    mkdir -p "$DIR"

    python -m coderl_lab.generation       --tasks "$DATA"       --output "$DIR/predictions.jsonl"       --metadata-output "$DIR/generation.json"       --model "$MODEL"       --revision "$REVISION"       --adapter "$ADAPTER"       --num-samples 1       --max-new-tokens 512       --greedy

    python -m coderl_lab.evaluation       --tasks "$DATA"       --predictions "$DIR/predictions.jsonl"       --output "$DIR/evaluation"       --executor docker       --timeout 5       --memory 512m       --max-workers 4       --k 1

    python -m coderl_lab.analysis.behavior_drift       --reference "$ROOT/greedy/sft.jsonl"       --candidate "$DIR/predictions.jsonl"       --output "$DIR/behavior_vs_sft.json"

    python -m coderl_lab.analysis.prompt_logit_sensitivity       --tasks "$DATA"       --model "$MODEL"       --revision "$REVISION"       --reference-adapter "$SFT"       --candidate-adapter "$ADAPTER"       --reference-cache "$CACHE"       --output "$ROOT/logits/$ARM.json"
  done
done
