#!/usr/bin/env bash
# TRAIN-013 full 237-step FP32-LoRA control, matched to completed TRAIN-012 BF16.
# Resume is intentionally NOT automatic; preserve checkpoint after failures.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv-gpu/bin/python}"
ADAPTER="${ADAPTER:-artifacts/train011/qlora-ultrachat}"
PREVIOUS="${PREVIOUS:-artifacts/train012/dpo_ultrafeedback_formal_trainonly_v1}"
CONFIG="configs/train013_dpo_fp32_full237.yaml"
OLD_CONFIG="configs/train012_7b_dpo_formal_trainonly_v1.yaml"
TRAIN="data/generated/posttrain-h4-v1/dpo_train.jsonl"
VAL="data/generated/posttrain-h4-v1/dpo_validation.jsonl"
MANIFEST="data/generated/posttrain-h4-v1/dpo_manifest.json"
OUT="artifacts/train013/fp32_full237"
test -x "$PYTHON"
test -s "$PREVIOUS/run_summary.json"
test ! -e "$OUT/training_identity.json" || { echo "Previous partial/full run exists; no implicit overwrite" >&2;exit 4; }
PYTHONPATH=src "$PYTHON" -m coderl_lab.train.train013_full237_freeze \
   --prior-config "$OLD_CONFIG" --new-config "$CONFIG" \
   --prior-summary "$PREVIOUS/run_summary.json" \
   --sft-adapter "$ADAPTER" --train "$TRAIN" \
   --heldout "$VAL" --manifest "$MANIFEST" \
   --output artifacts/train013/fp32_full237_cpu_freeze.json
(
   flock -n 9 || { echo "GPU experiment lock held" >&2;exit 9; }
   PYTHONPATH=src "$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 12000
   export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
   PYTHONPATH=src "$PYTHON" -m coderl_lab.train.dpo \
     --config "$CONFIG" --preferences "$TRAIN" \
     --sft-adapter "$ADAPTER" --output-dir "$OUT"
) 9>/tmp/coderl_lab_gpu_experiment_lock
echo "TRAIN-013 full237 FP32 precision-control GPU update and adapter save completed"
