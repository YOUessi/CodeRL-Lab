#!/usr/bin/env bash
# Formal DPO: new 4096-example SFT checkpoint and 2048 frozen preferences only.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
test -x "$PYTHON" || { echo "Python interpreter unavailable: $PYTHON" >&2; exit 2; }
# Use isolated bitsandbytes overlay without installing into the shared venv.
export PYTHONPATH="${QLORA_EXTRA_PYTHONPATH:+$QLORA_EXTRA_PYTHONPATH:}src${PYTHONPATH:+:$PYTHONPATH}"

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
RETRY_PRETRAINING_FAILURE="${RETRY_PRETRAINING_FAILURE:-0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
if [ -e "$OUTPUT/adapter_model.safetensors" ] || [ -e "$OUTPUT/run_summary.json" ]; then
  echo "Existing DPO final weights/summary; refusing overwrite" >&2
  exit 4
fi
if [ "$RETRY_PRETRAINING_FAILURE" != 0 ] && [ "$RETRY_PRETRAINING_FAILURE" != 1 ]; then
  echo "RETRY_PRETRAINING_FAILURE must be 0 or 1" >&2; exit 3
fi
if [ -n "$RESUME_CHECKPOINT" ] && [ "$RETRY_PRETRAINING_FAILURE" = 1 ]; then
  echo "Checkpoint resume and pretraining retry are exclusive" >&2; exit 3
fi
if [ -f "$EFFECTIVE_CONFIG" ] && [ -z "$RESUME_CHECKPOINT" ] &&
   [ "$RETRY_PRETRAINING_FAILURE" != 1 ]; then
  echo "Frozen DPO config exists; explicit resume or verified pretraining retry required" >&2; exit 3
fi
if [ ! -f "$EFFECTIVE_CONFIG" ] && { [ -n "$RESUME_CHECKPOINT" ] ||
   [ "$RETRY_PRETRAINING_FAILURE" = 1 ]; }; then
  echo "Frozen DPO config absent; cannot resume" >&2; exit 3
fi

# One file lock for all CodeRL-Lab GPU workloads. This formal run is not
# launched from CI and never terminates any other project's process.
(
  flock -n 9 || { echo "CodeRL-Lab GPU lock held" >&2; exit 9; }
  "$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 12000
  "$PYTHON" - <<'PY'
import bitsandbytes,torch
if not torch.cuda.is_available():
    raise RuntimeError("TRAIN-007C NF4 DPO requires real CUDA")
print("TRAIN-007C bnb/CUDA preflight",bitsandbytes.__version__,torch.cuda.get_device_name(0))
PY

  VERIFY_ARGS=()
  if [ -f "$EFFECTIVE_CONFIG" ]; then
    VERIFY_ARGS+=(--verify-existing)
  fi
  "$PYTHON" -m coderl_lab.train.posttrain_a_dpo_freeze \
    --adapter "$ADAPTER" \
    --sft-summary "$ADAPTER/run_summary.json" \
    --train-preferences "$DATA_TRAIN" \
    --validation-preferences "$DATA_EVAL" \
    --output "$EFFECTIVE_CONFIG" "${VERIFY_ARGS[@]}"

  TRAIN_ARGS=()
  if [ -n "$RESUME_CHECKPOINT" ]; then
    TRAIN_ARGS+=(--resume-from-checkpoint "$RESUME_CHECKPOINT")
  fi
  if [ "$RETRY_PRETRAINING_FAILURE" = 1 ]; then
    TRAIN_ARGS+=(--retry-pretraining-failure)
  fi
  "$PYTHON" -m coderl_lab.train.dpo \
    --config "$EFFECTIVE_CONFIG" \
    --preferences "$DATA_TRAIN" \
    --eval-preferences "$DATA_EVAL" \
    --sft-adapter "$ADAPTER" \
    --output-dir "$OUTPUT" "${TRAIN_ARGS[@]}"
) 9>/tmp/coderl_lab_gpu_experiment_lock

test -s "$OUTPUT/run_summary.json"
test -s "$OUTPUT/adapter_model.safetensors"
echo "TRAIN-007C full 2048/256 DPO completed: $OUTPUT/run_summary.json"
