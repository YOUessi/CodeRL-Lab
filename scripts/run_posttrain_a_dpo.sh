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

if [ -f "$EFFECTIVE_CONFIG" ]; then
  echo "Existing DPO frozen config; refusing unverified overwrite: $EFFECTIVE_CONFIG" >&2
  exit 3
fi
if [ -e "$OUTPUT/adapter_model.safetensors" ]; then
  echo "Existing trained DPO weights; refusing overwrite" >&2
  exit 4
fi

# Never produce an immutable frozen runtime config until GPU reservation passes.
# A failed preflight should leave this runner safely re-invocable without cleanup.
PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" \
  -m coderl_lab.train.gpu_preflight --min-free-mib 12000

PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" \
  -m coderl_lab.train.posttrain_a_dpo_freeze \
  --adapter "$ADAPTER" \
  --sft-summary "$ADAPTER/run_summary.json" \
  --train-preferences "$DATA_TRAIN" \
  --validation-preferences "$DATA_EVAL" \
  --output "$EFFECTIVE_CONFIG"

PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m coderl_lab.train.dpo \
  --config "$EFFECTIVE_CONFIG" \
  --preferences "$DATA_TRAIN" \
  --eval-preferences "$DATA_EVAL" \
  --sft-adapter "$ADAPTER" \
  --output-dir "$OUTPUT"

echo "TRAIN-007C DPO completed: $OUTPUT/run_summary.json"
