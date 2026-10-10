#!/usr/bin/env bash
# TRAIN-009A live-GPU on-policy RLHF. NOT PPO.
# MODE=smoke is fixed to 16 prompt-only rows, 4 samples/group, 1 optimizer step.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
MODE="${MODE:-smoke}"
A_WORKTREE="${A_WORKTREE:-/home/you/projects/CodeRL-Lab-track-a}"
RM_WORKTREE="${RM_WORKTREE:-/home/you/projects/CodeRL-Lab-track-rm}"
CONFIG="configs/train009a_learned_reward_grpo.yaml"
DATA_ROOT="$A_WORKTREE/data/generated/posttrain-h4-v1"
SFT_DIR="$A_WORKTREE/artifacts/posttrain-a/train007a-qlora-ultrachat"
REWARD_DIR="$RM_WORKTREE/artifacts/posttrain-a/train008a-reward-model"
ARGS=()
case "$MODE" in
  smoke)
    OUTPUT="artifacts/posttrain-a/train009a-reward-grpo-smoke-v2"
    ARGS+=(--smoke --max-train-prompts 16)
    ;;
  formal)
    OUTPUT="artifacts/posttrain-a/train009a-reward-grpo"
    # Prevent accidental unbudgeted full training. Formal budget requires
    # separate preregistration and positive max_steps in frozen config.
    "$PYTHON" - "$CONFIG" <<'PY'
import sys,yaml
c=yaml.safe_load(open(sys.argv[1]))
if int(c["training"]["max_steps"]) < 1:
    raise SystemExit("formal GRPO budget not frozen; no unbounded online RLHF")
PY
    ;;
  *) echo "MODE must be smoke or formal" >&2; exit 2 ;;
esac
test -x "$PYTHON" || { echo "Python interpreter missing" >&2; exit 2; }
export PYTHONPATH="${QLORA_EXTRA_PYTHONPATH:+$QLORA_EXTRA_PYTHONPATH:}src${PYTHONPATH:+:$PYTHONPATH}"
test ! -e "$OUTPUT/training_identity.json" || { echo "Existing online RLHF attempt: refusing overwrite" >&2; exit 3; }
test ! -e "$OUTPUT/run_summary.json" || { echo "Online model output already complete" >&2; exit 3; }

# First inspect exactly pinned actor, reward head and cross-domain prompts
# without reserving CUDA. No raw conversation text enters GitHub/logs.
"$PYTHON" - "$CONFIG" "$DATA_ROOT" "$SFT_DIR" "$REWARD_DIR" <<'PY'
from pathlib import Path
import json,sys,yaml
from coderl_lab.datasets.train009a_online_prompts import frozen_prompt_pool
from coderl_lab.train.learned_reward_grpo import validate_grpo_contract,validate_frozen_models
conf=Path(sys.argv[1]);data=Path(sys.argv[2])
sft=Path(sys.argv[3]);rm=Path(sys.argv[4])
c=yaml.safe_load(conf.read_text())
identity=validate_frozen_models(config=c,policy_dir=sft,reward_dir=rm)
contract=validate_grpo_contract(c,smoke=True)
rows,audit=frozen_prompt_pool(root=data,seed=42,max_train_prompts=16)
assert len(rows)==16 and not any("response" in row for row in rows)
print("TRAIN-009A PINNED preflight",json.dumps({
 "actor_sha":identity["initial_policy_sha256"],"rm_sha":identity["frozen_reward_sha256"],
 "eligible_disjoint_prompts":audit["eligible_train_prompts"],
 "smoke_prompts":len(rows),"KL_beta":contract["kl_beta"],
 "reward_is_frozen":True,"hidden_tests_used":False},ensure_ascii=False),flush=True)
PY
(
  flock -n 9 || { echo "CodeRL-Lab GPU experiment lock held" >&2; exit 9; }
  "$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 13500
  "$PYTHON" - <<'PY'
import torch,bitsandbytes,trl,peft
from trl import GRPOTrainer
assert torch.cuda.is_available() and torch.cuda.device_count()==1
assert not hasattr(trl,"PPOTrainer"), "Pinned TRL contract changed; review before use"
print("TRAIN-009A real CUDA+GRPO",torch.__version__,trl.__version__,bitsandbytes.__version__,flush=True)
PY
  "$PYTHON" -m coderl_lab.train.learned_reward_grpo \
    --config "$CONFIG" \
    --data-root "$DATA_ROOT" \
    --sft-dir "$SFT_DIR" --reward-dir "$REWARD_DIR" \
    --output-dir "$OUTPUT" "${ARGS[@]}"
) 9>/tmp/coderl_lab_gpu_experiment_lock

"$PYTHON" - "$OUTPUT" <<'PY'
import sys,json,hashlib,math
from pathlib import Path
root=Path(sys.argv[1]);p=root/"run_summary.json"
if not p.is_file():
    raise RuntimeError("No completed online RLHF summary")
r=json.loads(p.read_text())
a=root/"adapter_model.safetensors"
assert r["status"]=="smoke_only" and r["global_step"]==1
assert r["online_algorithm"].startswith("GRPO")
assert r["reference_policy"]["actual_pre_update_sft"] is True
assert r["reference_policy"]["ref_trainable_parameters"]==0
assert r["reward_audit"]["rewarded_rollouts"]>=4
assert r["new_policy_adapter_sha256"]==hashlib.sha256(a.read_bytes()).hexdigest()
assert math.isfinite(r["reward_audit"]["avg_bounded_reward_mean"])
print("TRAIN-009A real online RLHF smoke completed with verified frozen RM and SFT ref",flush=True)
PY
