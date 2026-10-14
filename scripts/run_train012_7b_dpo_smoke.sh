#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv-gpu/bin/python}"
CONFIG="configs/train012_7b_qlora_dpo_ultrafeedback.yaml"
SFT_ADAPTER="${SFT_ADAPTER:-artifacts/train011/qlora-ultrachat}"
TRAIN="data/generated/posttrain-h4-v1/dpo_train.jsonl"
VALID="data/generated/posttrain-h4-v1/dpo_validation.jsonl"
MANIFEST="data/generated/posttrain-h4-v1/dpo_manifest.json"
OUT="${OUT:-artifacts/train012/dpo_smoke_32pairs_2steps}"
test -x "$PYTHON"
test ! -e "$OUT/run_summary.json" || { echo "Existing finished DPO smoke; refusing overwrite" >&2; exit 4; }
test ! -e "$OUT/training_identity.json" || { echo "Existing partial DPO run; refusing implicit resume" >&2; exit 4; }
PYTHONPATH=src "$PYTHON" -m coderl_lab.train.train012_preflight \
  --config "$CONFIG" --sft-adapter "$SFT_ADAPTER" \
  --dpo-train "$TRAIN" --dpo-validation "$VALID" --dpo-manifest "$MANIFEST" \
  --output artifacts/train012/preflight_verified.json
(
  flock -n 9 || { echo "GPU experiment lock held" >&2; exit 9; }
  PYTHONPATH=src "$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 12000
  PYTHONPATH=src "$PYTHON" -m coderl_lab.train.dpo \
    --config "$CONFIG" \
    --preferences "$TRAIN" \
    --eval-preferences "$VALID" \
    --sft-adapter "$SFT_ADAPTER" \
    --output-dir "$OUT" \
    --max-pairs 32 --max-eval-pairs 8 --max-steps 2
) 9>/tmp/coderl_lab_gpu_experiment_lock
echo "TRAIN-012 real 7B DPO two-step CUDA smoke completed"
