#!/usr/bin/env bash
# Post-training TRAIN-013 full237 paired BF16 vs FP32 assessment.
# Run only after full237 FP32 Adapter and summary are verifiably saved.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv-gpu/bin/python}"
SFT="${SFT:-artifacts/train011/qlora-ultrachat}"
PRIOR="${PRIOR:-artifacts/train012/dpo_ultrafeedback_formal_trainonly_v1}"
FP32="artifacts/train013/fp32_full237"
FREEZE="artifacts/train013/fp32_full237_cpu_freeze.json"
AUDIT="artifacts/train013/fp32_full237_parameter_audit.json"
VALID="data/generated/posttrain-h4-v1/dpo_validation.jsonl"
test -x "$PYTHON"
test -s "$FP32/run_summary.json" || { echo "FP32 full237 training incomplete";exit 3; }
test -s "$PRIOR/run_summary.json"
test -s "$FREEZE"
PYTHONPATH=src "$PYTHON" -m coderl_lab.analysis.train013_full237_parameter_audit \
    --sft "$SFT/adapter_model.safetensors" \
    --prior-bf16 "$PRIOR" --new-fp32 "$FP32" \
    --prior-config configs/train012_7b_dpo_formal_trainonly_v1.yaml \
    --new-config configs/train013_dpo_fp32_full237.yaml \
    --freeze "$FREEZE" --out "$AUDIT"
(
  flock -n 9 || { echo "GPU lock held";exit 9; }
  PYTHONPATH=src "$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 10000
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  for quantity in 8 256; do
    OUT="artifacts/train013/full237_heldout_$quantity"
    if [ -e "$OUT/summary.json" ]; then
      echo "Output $OUT already complete: do not rescore"
      continue
    fi
    PYTHONPATH=src "$PYTHON" -m coderl_lab.analysis.train013_full237_heldout \
      --heldout "$VALID" --sft "$SFT" --bf16 "$PRIOR" \
      --fp32 "$FP32" --report "$FREEZE" \
      --output-dir "$OUT" --max-pairs "$quantity"
  done
) 9>/tmp/coderl_lab_gpu_experiment_lock
echo "TRAIN-013 FP32 full237 and retrospective 256-pair validation completed"
