#!/usr/bin/env bash
# TRAIN-007D: real CUDA, frozen SFT vs DPO 252-pair likelihood comparison.
# No training and no retuning. All tasks in exact 256-pair H4 heldout.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
A_WORKTREE="${A_WORKTREE:-/home/you/projects/CodeRL-Lab-track-a}"
ROOT="${ROOT:-artifacts/train007d-matched-preference}"
PREFERENCES="$A_WORKTREE/data/generated/posttrain-h4-v1"
SOURCE_SFT="$A_WORKTREE/artifacts/posttrain-a/train007a-qlora-ultrachat"
SOURCE_DPO="$A_WORKTREE/artifacts/posttrain-a/train007c-dpo-ultrafeedback"

test -x "$PYTHON" || { echo "Required python interpreter unavailable: $PYTHON" >&2; exit 2; }
test -s "$SOURCE_SFT/adapter_model.safetensors"
test -s "$SOURCE_DPO/adapter_model.safetensors"
test -s "$PREFERENCES/dpo_validation.jsonl"
export PYTHONPATH="${QLORA_EXTRA_PYTHONPATH:+$QLORA_EXTRA_PYTHONPATH:}src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" - "$SOURCE_SFT" "$SOURCE_DPO" "$PREFERENCES" <<'PY'
from pathlib import Path
import sys
from coderl_lab.analysis.train007d_preference_eval import verify_frozen_sources
sft,dpo,data=map(Path,sys.argv[1:])
f=verify_frozen_sources(
 manifest_path=data/"dpo_manifest.json",
 heldout_path=data/"dpo_validation.jsonl",
 train_path=data/"dpo_train.jsonl",
 sft_adapter=sft/"adapter_model.safetensors",
 dpo_adapter=dpo/"adapter_model.safetensors",
 sft_summary=sft/"run_summary.json",
 dpo_summary=dpo/"run_summary.json",
)
assert f["no_optimizer_training"] is True
assert f["no_private_or_hidden_tests"] is True
print("Pinned SFT, full DPO and H4 heldout adapter/source SHA match",flush=True)
PY
(
  flock -n 9 || { echo "GPU experiment lock occupied; no evaluation launched" >&2; exit 9; }
  "$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 12000
  "$PYTHON" - <<'PY'
import torch,bitsandbytes,peft
assert torch.cuda.is_available()
print("Pinned NF4 matched evaluation CUDA:",bitsandbytes.__version__,flush=True)
PY
  "$PYTHON" -m coderl_lab.analysis.train007d_preference_eval \
    --root "$ROOT" \
    --sft-adapter "$SOURCE_SFT" \
    --dpo-adapter "$SOURCE_DPO" \
    --sft-summary "$SOURCE_SFT/run_summary.json" \
    --dpo-summary "$SOURCE_DPO/run_summary.json" \
    --train "$PREFERENCES/dpo_train.jsonl" \
    --heldout "$PREFERENCES/dpo_validation.jsonl" \
    --manifest "$PREFERENCES/dpo_manifest.json"
) 9>/tmp/coderl_lab_gpu_experiment_lock

test -s "$ROOT/summary.json"
echo "TRAIN-007D all 252 paired heldout results frozen: $ROOT/summary.json"
