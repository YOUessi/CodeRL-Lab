#!/usr/bin/env bash
# EXP-014: score 256 independent HH preferences after 192 unlabeled responses.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv-gpu/bin/python}"
ROOT="${ROOT:-artifacts/exp014}"
SFT="${SFT:-artifacts/train011/qlora-ultrachat}"
BF16="${BF16:-artifacts/train012/dpo_ultrafeedback_formal_trainonly_v1}"
FP32="${FP32:-artifacts/train013/fp32_full237}"
test -x "$PYTHON"
test -s "$ROOT/blind64/summary.json" || { echo "192 answers must freeze BEFORE human labels are opened" >&2;exit 4; }
if [ -s "$ROOT/preference256/summary.json" ];then echo "Already scored 256 HH pairs; refusing overwrite" >&2;exit 3;fi
(
 flock -n 9 || { echo "Another Tang GPU study holds lock" >&2;exit 9; }
 PYTHONPATH=src "$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 11000
 export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
 PYTHONPATH=src "$PYTHON" -m coderl_lab.analysis.exp014_hh_preference \
   --public-prompts "$ROOT/frozen256/public_prompts.jsonl" \
   --private-pairs "$ROOT/frozen256/private_pairs.jsonl" \
   --manifest "$ROOT/frozen256/manifest.json" \
   --generation-summary "$ROOT/blind64/summary.json" \
   --sft "$SFT" --bf16-dpo "$BF16" --fp32-dpo "$FP32" \
   --output-dir "$ROOT/preference256"
) 9>/tmp/coderl_lab_gpu_experiment_lock
echo "EXP014 independent HH 256 preference pairs score complete"
