#!/usr/bin/env bash
# TRAIN-016: Qwen2.5-7B-Instruct + frozen TRAIN-011 SFT LoRA,
# actual quantized GRPO with Docker execution rewards.
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE="${1:?smoke|formal}"
if [ "$STAGE" != smoke ] && [ "$STAGE" != formal ]; then
  echo "Expected smoke or formal" >&2
  exit 2
fi
PYTHON="${PYTHON:-/home/you/projects/CodeRL-Lab/.venv-gpu/bin/python}"
SFT_ADAPTER="${SFT_ADAPTER:-artifacts/train011/qlora-ultrachat}"
CONFIG=configs/train016_7b_qlora_grpo.yaml
DATA_DIR="${DATA_DIR:-artifacts/train014b/frozen_data}"
OUTPUT="artifacts/train016/grpo7b_${STAGE}_v1"
STEPS=2
[ "$STAGE" = formal ] && STEPS=16
test -x "$PYTHON"
test -s "$SFT_ADAPTER/adapter_model.safetensors"
test -s "$DATA_DIR/manifest.json"
if [ -e "$OUTPUT/identity.json" ] || [ -e "$OUTPUT/summary.json" ];then
 echo "TRAIN-016 $STAGE exists: refusing overwrite/unsafe resume" >&2
 exit 4
fi
(
 flock -n 9 || { echo "GPU already locked by another project" >&2; exit 9; }
 export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
 PYTHONPATH=src "$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 12000
 PYTHONPATH=src "$PYTHON" -m coderl_lab.train.train016_7b_qlora_grpo \
   --config "$CONFIG" --data-dir "$DATA_DIR" \
   --sft-adapter "$SFT_ADAPTER" \
   --output-dir "$OUTPUT" --max-steps "$STEPS" --run-kind "$STAGE"
) 9>/tmp/coderl_lab_gpu_experiment_lock
echo "TRAIN016 7B SFT-continuation actual GRPO $STAGE complete"
