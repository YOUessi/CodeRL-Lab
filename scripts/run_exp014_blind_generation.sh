#!/usr/bin/env bash
# EXP-014: freeze all64 public-only HH prompts into 192 blinded candidate outputs.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv-gpu/bin/python}"
ROOT="${ROOT:-artifacts/exp014}"
SFT="${SFT:-artifacts/train011/qlora-ultrachat}"
BF16="${BF16:-artifacts/train012/dpo_ultrafeedback_formal_trainonly_v1}"
FP32="${FP32:-artifacts/train013/fp32_full237}"
if [ -e "$ROOT/blind64/summary.json" ];then echo "Completed cohort immutable" >&2;exit 3;fi
test -x "$PYTHON"
test -s "$ROOT/frozen256/manifest.json"
(
 flock -n 9 || { echo "Another Tang GPU study holds the lock" >&2;exit 9; }
 PYTHONPATH=src "$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 11000
 export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
 PYTHONPATH=src "$PYTHON" -m coderl_lab.analysis.exp014_blind_generate \
   --public-prompts "$ROOT/frozen256/generation_prompts.jsonl" \
   --manifest "$ROOT/frozen256/manifest.json" \
   --sft "$SFT" --bf16-dpo "$BF16" --fp32-dpo "$FP32" \
   --output-dir "$ROOT/blind64"
) 9>/tmp/coderl_lab_gpu_experiment_lock
echo "EXP014 full64 generation completed and blind reviewer packets frozen"
