#!/usr/bin/env bash
# EXP-014B: posthoc engineering repair of 224-token truncation.
# Same frozen 64 HH prompts and model seeds; new max 512 and reviewer blind order.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv-gpu/bin/python}"
SOURCE="${SOURCE:-artifacts/exp014/frozen256}"
SFT="${SFT:-artifacts/train011/qlora-ultrachat}"
BF16="${BF16:-artifacts/train012/dpo_ultrafeedback_formal_trainonly_v1}"
FP32="${FP32:-artifacts/train013/fp32_full237}"
OUT="${OUT:-artifacts/exp014b/long64}"
test -x "$PYTHON"
test -s "$SOURCE/manifest.json"
test ! -s "$OUT/summary.json" || { echo "Long review packet already frozen";exit 4; }
(
 flock -n 9 || { echo "Another Tang GPU task is active" >&2;exit 9; }
 PYTHONPATH=src "$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 11000
 export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
 PYTHONPATH=src "$PYTHON" -m coderl_lab.analysis.exp014b_blind_generate \
  --public-prompts "$SOURCE/generation_prompts.jsonl" \
  --manifest "$SOURCE/manifest.json" \
  --sft "$SFT" --bf16-dpo "$BF16" --fp32-dpo "$FP32" \
  --output-dir "$OUT"
) 9>/tmp/coderl_lab_gpu_experiment_lock
echo "EXP014B long reply reviewed dataset generation finished, HUMAN SCORES STILL REQUIRED"
