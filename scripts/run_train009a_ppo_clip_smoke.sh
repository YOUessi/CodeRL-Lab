#!/usr/bin/env bash
# TRAIN-009A: one GPU, frozen RM+SFT, genuine on-policy PPO-Clip smoke.
# Never claim this is a validated human RLHF improvement or full PPO benchmark.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/you/projects/CodeRL-Lab-track-a/data/generated/posttrain-h4-v1}"
SFT_DIR="${SFT_DIR:-/home/you/projects/CodeRL-Lab-track-a/artifacts/posttrain-a/train007a-qlora-ultrachat}"
RM_DIR="${RM_DIR:-/home/you/projects/CodeRL-Lab-track-rm/artifacts/posttrain-a/train008a-reward-model}"
CONFIG="configs/train009a_ppo_clip_rlhf_smoke.yaml"
OUTPUT="artifacts/posttrain-a/train009a-ppo-clip-smoke"
export PYTHONPATH="${QLORA_EXTRA_PYTHONPATH:+$QLORA_EXTRA_PYTHONPATH:}src${PYTHONPATH:+:$PYTHONPATH}"

test -x "$PYTHON" || { echo "Pinned Python venv missing" >&2; exit 2; }
test ! -e "$OUTPUT/training_identity.json" || {
  echo "Existing PPO run may contain training state; refusing overwrite" >&2; exit 3;
}
test ! -e "$OUTPUT/run_summary.json" || {
  echo "Completed PPO smoke exists, refusing duplicate" >&2; exit 3;
}

# Hash all actual model weights and data before getting a CUDA lock.
"$PYTHON" - "$CONFIG" "$DATA_ROOT" "$SFT_DIR" "$RM_DIR" <<'PY'
from pathlib import Path
import sys
from coderl_lab.train.ppo_online import frozen_provenance
config,data,sft,rm=map(Path,sys.argv[1:])
rollouts,probes,identity=frozen_provenance(
    config_path=config,data_root=data,sft_dir=sft,reward_dir=rm,
)
assert len(rollouts)==8 and len(probes)==4
assert identity["frozen_data"]["selection_overlaps_any_heldout"]==0
assert identity["frozen_data"]["selection_overlaps_reward_training"]==0
assert identity["no_heldout_preference_labels_used_for_policy_gradient"] is True
print("TRAIN-009A frozen SFT+RM adapter SHA, 8 rollout + 4 probe prompts, and heldout disjointness PASS",flush=True)
PY

(
  flock -n 9 || { echo "GPU experiment lock occupied" >&2; exit 9; }
  "$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 11000
  "$PYTHON" - <<'PY'
import torch,bitsandbytes,peft
assert torch.cuda.is_available()
print("PPO GPU/isolated bitsandbytes:",torch.cuda.get_device_name(0),bitsandbytes.__version__,flush=True)
PY
  "$PYTHON" -m coderl_lab.train.ppo_online \
    --config "$CONFIG" \
    --data-root "$DATA_ROOT" \
    --sft-adapter "$SFT_DIR" \
    --reward-adapter "$RM_DIR" \
    --output-dir "$OUTPUT"
) 9>/tmp/coderl_lab_gpu_experiment_lock

"$PYTHON" - "$OUTPUT" <<'PY'
import json,hashlib,sys,math
from pathlib import Path
root=Path(sys.argv[1])
r=json.loads((root/"run_summary.json").read_text())
assert r["experiment"]=="TRAIN-009A"
assert r["status"]=="real_cuda_ppo_clip_smoke_not_formal"
assert r["steps"]==4 and r["training_episodes"]==16
assert r["actual_optimizer_minibatch_steps"]==32
assert r["source_identity"]["no_heldout_preference_labels_used_for_policy_gradient"]
assert r["source_identity"]["no_private_or_hidden_tests"]
assert math.isfinite(r["mean_sampled_on_policy_KL_to_SFT_reference"])
w=Path(r["policy_adapter_path"])
assert w.is_file()
assert hashlib.sha256(w.read_bytes()).hexdigest()==r["policy_adapter_sha256"]
assert hashlib.sha256((root/"value_head.safetensors").read_bytes()).hexdigest()==r["value_head_sha256"]
assert hashlib.sha256((root/"optimizer_state.pt").read_bytes()).hexdigest()==r["optimizer_state_sha256"]
print("TRAIN-009A 4step/16 episode actual PPO smoke adapter/value/optimizer SHA integrity PASS",flush=True)
PY
