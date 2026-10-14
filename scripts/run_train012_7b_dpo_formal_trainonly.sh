#!/usr/bin/env bash
# TRAIN-012 full UltraFeedback 2048 DPO steps from verified TRAIN-011 SFT.
# Separate heldout validation avoids the already reproduced Trainer FP32 logits OOM.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv-gpu/bin/python}"
SFT_ADAPTER="${SFT_ADAPTER:-artifacts/train011/qlora-ultrachat}"
CONFIG="configs/train012_7b_dpo_formal_trainonly_v1.yaml"
TRAIN="data/generated/posttrain-h4-v1/dpo_train.jsonl"
VALID="data/generated/posttrain-h4-v1/dpo_validation.jsonl"
MANIFEST="data/generated/posttrain-h4-v1/dpo_manifest.json"
OUT="artifacts/train012/dpo_ultrafeedback_formal_trainonly_v1"
test -x "$PYTHON"
if [ -e "$OUT/training_identity.json" ] || [ -e "$OUT/adapter_model.safetensors" ]; then
  echo "Formal output/identity already exists: do not implicitly resume or overwrite" >&2
  exit 4
fi
PYTHONPATH=src "$PYTHON" -m coderl_lab.train.train012_preflight \
  --config "$CONFIG" --sft-adapter "$SFT_ADAPTER" \
  --dpo-train "$TRAIN" --dpo-validation "$VALID" --dpo-manifest "$MANIFEST" \
  --output artifacts/train012/preflight_verified_formal_v1.json
(
  flock -n 9 || { echo "GPU experiment lock held" >&2; exit 9; }
  PYTHONPATH=src "$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 12000
  PYTHONPATH=src "$PYTHON" -m coderl_lab.train.dpo \
    --config "$CONFIG" \
    --preferences "$TRAIN" \
    --sft-adapter "$SFT_ADAPTER" \
    --output-dir "$OUT"
) 9>/tmp/coderl_lab_gpu_experiment_lock
echo "TRAIN-012 full frozen 2048-pair DPO formal training completed"
