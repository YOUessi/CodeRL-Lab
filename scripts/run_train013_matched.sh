#!/usr/bin/env bash
# TRAIN-013: matched BF16/FP32 7B DPO real GPU experiment on Tang.
# Usage: bash scripts/run_train013_matched.sh bf16|fp32 smoke|formal
set -euo pipefail
cd "$(dirname "$0")/.."
PRECISION="${1:?bf16|fp32 required}"
STAGE="${2:?smoke|formal required}"
case "$PRECISION" in bf16|fp32) ;; *) echo "unsupported precision" >&2; exit 2;; esac
case "$STAGE" in smoke|formal) ;; *) echo "unsupported stage" >&2; exit 2;; esac
PYTHON="${PYTHON:-.venv-gpu/bin/python}"
ADAPTER="${ADAPTER:-artifacts/train011/qlora-ultrachat}"
CONFIG="configs/train013_dpo_7b_matched32_${PRECISION}.yaml"
TRAIN="data/generated/posttrain-h4-v1/dpo_train.jsonl"
VAL="data/generated/posttrain-h4-v1/dpo_validation.jsonl"
MANIFEST="data/generated/posttrain-h4-v1/dpo_manifest.json"
OUT="artifacts/train013/${STAGE}_${PRECISION}"
test -x "$PYTHON"
if [ -f "$OUT/training_identity.json" ] || [ -f "$OUT/run_summary.json" ]; then
  echo "Existing run: refusing implicit overwrite/restart" >&2
  exit 4
fi
PYTHONPATH=src "$PYTHON" -m coderl_lab.train.train013_freeze \
  --bf16 configs/train013_dpo_7b_matched32_bf16.yaml \
  --fp32 configs/train013_dpo_7b_matched32_fp32.yaml \
  --adapter "$ADAPTER" --train "$TRAIN" --heldout "$VAL" \
  --manifest "$MANIFEST" --output artifacts/train013/paired_cpu_freeze.json
EXTRA=()
if [ "$STAGE" = smoke ]; then
    EXTRA+=(--max-pairs 32 --max-steps 2)
fi
(
  flock -n 9 || { echo "GPU exclusive lock held" >&2; exit 9; }
  PYTHONPATH=src "$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 12000
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  PYTHONPATH=src "$PYTHON" -m coderl_lab.train.dpo \
    --config "$CONFIG" --preferences "$TRAIN" \
    --sft-adapter "$ADAPTER" --output-dir "$OUT" \
    "${EXTRA[@]}"
) 9>/tmp/coderl_lab_gpu_experiment_lock
echo "TRAIN-013 ${PRECISION} ${STAGE} completed after real GPU optimizer+adapter-save"
