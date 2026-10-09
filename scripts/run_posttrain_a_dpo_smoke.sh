#!/usr/bin/env bash
# TRAIN-007C smoke uses 32 real frozen UltraFeedback training pairs + 16 heldout pairs.
# Not the formal 2048/256 run; no claim of PPO/RLHF completion.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
# Keep bitsandbytes in an isolated read-only Python overlay. No global pip
# install or unrelated conda environment changes.
export PYTHONPATH="${QLORA_EXTRA_PYTHONPATH:+$QLORA_EXTRA_PYTHONPATH:}src${PYTHONPATH:+:$PYTHONPATH}"

ADAPTER="artifacts/posttrain-a/train007a-qlora-ultrachat"
DATA="data/generated/posttrain-h4-v1"
OUTPUT="artifacts/posttrain-a/train007c-dpo-smoke"
FROZEN="$OUTPUT/frozen_config.yaml"

test -x "$PYTHON" || { echo "Python interpreter is missing" >&2; exit 2; }
RETRY_PRETRAINING_FAILURE="${RETRY_PRETRAINING_FAILURE:-0}"
ARGS=()
if [ "$RETRY_PRETRAINING_FAILURE" = "1" ]; then
  # Strong allow-list, later cross-checked against immutable hashes by Python.
  test -f "$FROZEN" && test -f "$OUTPUT/training_identity.json" || {
    echo "DPO frozen config/identity missing; cannot retry" >&2; exit 3;
  }
  ARGS+=(--retry-pretraining-failure)
elif [ "$RETRY_PRETRAINING_FAILURE" = "0" ]; then
  if [ -e "$FROZEN" ] || [ -e "$OUTPUT/training_identity.json" ] ||
     [ -e "$OUTPUT/run_summary.json" ] || [ -e "$OUTPUT/adapter_model.safetensors" ]; then
    echo "Existing DPO smoke output; use explicit RETRY_PRETRAINING_FAILURE=1 for immutable validation" >&2
    exit 3
  fi
else
  echo "RETRY_PRETRAINING_FAILURE must be 0 or 1" >&2
  exit 3
fi

(
  flock -n 9 || { echo "CodeRL-Lab GPU lock held; refusing parallel DPO" >&2; exit 9; }
  "$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 12000
  "$PYTHON" - <<'PY'
import torch
import bitsandbytes
assert torch.cuda.is_available()
print("NF4 DPO isolated dependency verified:", bitsandbytes.__version__, flush=True)
PY

  # Full source check is mandatory even for a small GPU smoke subset.
  VERIFY_ARGS=()
  if [ "$RETRY_PRETRAINING_FAILURE" = "1" ]; then
    VERIFY_ARGS+=(--verify-existing)
  fi
  "$PYTHON" -m coderl_lab.train.posttrain_a_dpo_freeze \
    --adapter "$ADAPTER" \
    --sft-summary "$ADAPTER/run_summary.json" \
    --manifest "$DATA/dpo_manifest.json" \
    --train-preferences "$DATA/dpo_train.jsonl" \
    --validation-preferences "$DATA/dpo_validation.jsonl" \
    --output "$FROZEN" "${VERIFY_ARGS[@]}"

  "$PYTHON" -m coderl_lab.train.dpo \
    --config "$FROZEN" \
    --preferences "$DATA/dpo_train.jsonl" \
    --eval-preferences "$DATA/dpo_validation.jsonl" \
    --sft-adapter "$ADAPTER" \
    --output-dir "$OUTPUT" \
    --max-pairs 32 --max-eval-pairs 16 --max-steps 2 "${ARGS[@]}"

  "$PYTHON" - "$OUTPUT" <<'PY'
import hashlib,json,math,sys
from pathlib import Path
root=Path(sys.argv[1])
x=json.loads((root/"run_summary.json").read_text())
assert x["cuda_available"] is True
assert x["num_pairs"]==32 and x["num_heldout_preference_pairs"]==16
assert x["global_step"]==2
assert x["quantization"]["mode"]=="nf4"
assert x["checkpoint_policy"]["save_strategy"]=="steps"
assert math.isfinite(x["metrics"]["train_loss"])
assert (root/"adapter_model.safetensors").exists()
assert x["output_adapter_sha256"]==hashlib.sha256(
    (root/"adapter_model.safetensors").read_bytes()).hexdigest()
history=json.loads((root/"log_history.json").read_text())
assert any(any(k.startswith("eval_") or k.startswith("eval/") for k in row)
           for row in history)
print("TRAIN-007C real DPO 32/16 two-step CUDA smoke PASS",flush=True)
PY
) 9>/tmp/coderl_lab_gpu_experiment_lock
