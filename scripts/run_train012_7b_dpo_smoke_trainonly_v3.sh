#!/usr/bin/env bash
# TRAIN-012 v3 two-step DPO optimizer and Adapter-save smoke ONLY.
# Frozen 256 heldout preference pairs are checked by SHA, not loaded into
# DPOTrainer validation to avoid evaluation's FP32 full-logit GPU OOM.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv-gpu/bin/python}"
CONFIG="configs/train012_7b_dpo_smoke_train_only_v3.yaml"
SFT_ADAPTER="${SFT_ADAPTER:-artifacts/train011/qlora-ultrachat}"
TRAIN="data/generated/posttrain-h4-v1/dpo_train.jsonl"
VALID="data/generated/posttrain-h4-v1/dpo_validation.jsonl"
MANIFEST="data/generated/posttrain-h4-v1/dpo_manifest.json"
OUT="artifacts/train012/dpo_smoke_32pairs_2steps_trainonly_v3"
test -x "$PYTHON"
test ! -e "$OUT/training_identity.json" || { echo "Existing partial DPO v3 experiment; no implicit overwrite" >&2; exit 4; }
PYTHONPATH=src "$PYTHON" -m coderl_lab.train.train012_preflight \
  --config "$CONFIG" --sft-adapter "$SFT_ADAPTER" \
  --dpo-train "$TRAIN" --dpo-validation "$VALID" --dpo-manifest "$MANIFEST" \
  --output artifacts/train012/preflight_verified_trainonly_v3.json
(
  flock -n 9 || { echo "GPU experiment lock held" >&2; exit 9; }
  PYTHONPATH=src "$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 12000
  PYTHONPATH=src "$PYTHON" -m coderl_lab.train.dpo \
    --config "$CONFIG" --preferences "$TRAIN" \
    --sft-adapter "$SFT_ADAPTER" --output-dir "$OUT" \
    --max-pairs 32 --max-steps 2
) 9>/tmp/coderl_lab_gpu_experiment_lock
echo "TRAIN-012 DPO two optimizer steps and Adapter save completed"
