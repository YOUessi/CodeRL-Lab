#!/usr/bin/env bash
# TRAIN-013 paired32 heldout diagnostic with exact frozen SFT and DPO adapters.
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE="${1:?smoke|formal required}"
case "$STAGE" in smoke|formal) ;; *) echo "Expected smoke|formal" >&2;exit 2;; esac
PYTHON="${PYTHON:-.venv-gpu/bin/python}"
BASE="${BASE:-artifacts/train013}"
SFT="${SFT:-artifacts/train011/qlora-ultrachat}"
VALID="data/generated/posttrain-h4-v1/dpo_validation.jsonl"
REPORT="$BASE/precision_report.json"
BF16="$BASE/formal_bf16"
FP32="$BASE/formal_fp32"
MAX=256
OUT="$BASE/heldout_formal256"
if [ "$STAGE" = smoke ]; then MAX=8; OUT="$BASE/heldout_smoke8";fi
test -x "$PYTHON"
test -s "$REPORT"
test -s "$BF16/run_summary.json"
test -s "$FP32/run_summary.json"
test -s "$VALID"
test -s "$SFT/adapter_model.safetensors"
test ! -e "$OUT/summary.json" || { echo "Already finalized, refusing overwrite" >&2; exit 3; }
(
  flock -n 9 || { echo "GPU experiment locked by another task" >&2; exit 9; }
  PYTHONPATH=src "$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 10000
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  PYTHONPATH=src "$PYTHON" -m coderl_lab.analysis.train013_heldout_preference \
    --heldout "$VALID" --sft "$SFT" \
    --bf16 "$BF16" --fp32 "$FP32" \
    --report "$REPORT" --output-dir "$OUT" --max-pairs "$MAX"
) 9>/tmp/coderl_lab_gpu_experiment_lock
echo "TRAIN-013 matched32 $STAGE heldout diagnostics completed"
