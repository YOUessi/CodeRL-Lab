#!/usr/bin/env bash
# Formal DPO: new 4096-example SFT checkpoint and 2048 frozen preferences only.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
test -x "$PYTHON" || { echo "Python interpreter unavailable: $PYTHON" >&2; exit 2; }

ADAPTER="${ADAPTER:-artifacts/posttrain-a/train007a-qlora-ultrachat}"
OUTPUT="${OUTPUT:-artifacts/posttrain-a/train007c-dpo-ultrafeedback}"
EFFECTIVE_CONFIG="$OUTPUT/frozen_config.yaml"
DATA_TRAIN="data/generated/posttrain-h4-v1/dpo_train.jsonl"
DATA_EVAL="data/generated/posttrain-h4-v1/dpo_validation.jsonl"

RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
if [ -e "$OUTPUT/adapter_model.safetensors" ] || [ -e "$OUTPUT/run_summary.json" ]; then
  echo "Existing DPO final output; refusing overwrite" >&2
  exit 4
fi
if [ -f "$EFFECTIVE_CONFIG" ] && [ -z "$RESUME_CHECKPOINT" ]; then
  echo "Frozen DPO config already exists; provide explicit RESUME_CHECKPOINT" >&2
  exit 3
fi
if [ ! -f "$EFFECTIVE_CONFIG" ] && [ -n "$RESUME_CHECKPOINT" ]; then
  echo "DPO freeze file absent; cannot safely resume" >&2
  exit 3
fi

# Enforce GPU exclusivity against both A and B experiments.
(
  flock -n 9 || { echo "CodeRL-Lab GPU experiment lock held" >&2; exit 9; }
  PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" \
    -m coderl_lab.train.gpu_preflight --min-free-mib 12000

  if [ ! -f "$EFFECTIVE_CONFIG" ]; then
    PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" \
      -m coderl_lab.train.posttrain_a_dpo_freeze \
      --adapter "$ADAPTER" \
      --sft-summary "$ADAPTER/run_summary.json" \
      --train-preferences "$DATA_TRAIN" \
      --validation-preferences "$DATA_EVAL" \
      --output "$EFFECTIVE_CONFIG"
  fi

  ARGS=()
  if [ -n "$RESUME_CHECKPOINT" ]; then
    ARGS+=(--resume-from-checkpoint "$RESUME_CHECKPOINT")
  fi
  PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" \
    -m coderl_lab.train.dpo \
    --config "$EFFECTIVE_CONFIG" \
    --preferences "$DATA_TRAIN" \
    --eval-preferences "$DATA_EVAL" \
    --sft-adapter "$ADAPTER" \
    --output-dir "$OUTPUT" "${ARGS[@]}"
) 9>/tmp/coderl_lab_gpu_experiment_lock

echo "TRAIN-007C DPO completed: $OUTPUT/run_summary.json"
