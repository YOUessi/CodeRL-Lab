#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-artifacts/exp004f/smoke}"
DATA="${DATA:-data/generated/mbpp-v1/validation_tasks.jsonl}"
MODEL="Qwen/Qwen3-1.7B-Base"
REVISION="ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
SFT="${SFT:-artifacts/exp006a/sft-qwen3-1.7b-lora}"
CAND="${CAND:-artifacts/exp004e/perturbations/scale-025/seed-202}"

rm -rf "$ROOT"
mkdir -p "$ROOT"

python -m coderl_lab.generation   --tasks "$DATA"   --output "$ROOT/sft_greedy.jsonl"   --metadata-output "$ROOT/sft_greedy_generation.json"   --model "$MODEL"   --revision "$REVISION"   --adapter "$SFT"   --max-tasks 3   --num-samples 1   --max-new-tokens 256   --greedy

python -m coderl_lab.generation   --tasks "$DATA"   --output "$ROOT/candidate_greedy.jsonl"   --metadata-output "$ROOT/candidate_greedy_generation.json"   --model "$MODEL"   --revision "$REVISION"   --adapter "$CAND"   --max-tasks 3   --num-samples 1   --max-new-tokens 256   --greedy

python -m coderl_lab.analysis.behavior_drift   --reference "$ROOT/sft_greedy.jsonl"   --candidate "$ROOT/candidate_greedy.jsonl"   --output "$ROOT/greedy_behavior_vs_sft.json"

python -m coderl_lab.analysis.prompt_logit_sensitivity   --tasks "$DATA"   --model "$MODEL"   --revision "$REVISION"   --reference-adapter "$SFT"   --candidate-adapter "$CAND"   --reference-cache "$ROOT/reference_prompt_logits.pt"   --output "$ROOT/prompt_logit_sensitivity.json"   --max-tasks 3

python -m coderl_lab.evaluation   --tasks "$DATA"   --predictions "$ROOT/sft_greedy.jsonl"   --output "$ROOT/sft_eval"   --executor docker   --timeout 5   --memory 512m   --max-workers 2   --k 1

python -m coderl_lab.evaluation   --tasks "$DATA"   --predictions "$ROOT/candidate_greedy.jsonl"   --output "$ROOT/candidate_eval"   --executor docker   --timeout 5   --memory 512m   --max-workers 2   --k 1
