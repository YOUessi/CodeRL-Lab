#!/usr/bin/env bash
# TRAIN-009C: natural-EOS guarded GRPO 512-token smoke; original SFT/RM/data.
# Unlike A v3, this test DOES mask all unfinished completions and applies
# learned RM only to naturally terminated completions. No PPO claim.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
DATA_ROOT="${A_WORKTREE:-/home/you/projects/CodeRL-Lab-track-a}/data/generated/posttrain-h4-v1"
SFT_DIR="${A_WORKTREE:-/home/you/projects/CodeRL-Lab-track-a}/artifacts/posttrain-a/train007a-qlora-ultrachat"
RM_DIR="${RM_WORKTREE:-/home/you/projects/CodeRL-Lab-track-rm}/artifacts/posttrain-a/train008a-reward-model"
CONFIG="configs/train009c_eos_guarded_grpo_smoke.yaml"
OUTPUT="artifacts/posttrain-a/train009c-eos-guarded-grpo-smoke"

test -x "$PYTHON" || { echo "TRAIN009C interpreter missing" >&2; exit 2; }
test ! -e "$OUTPUT/training_identity.json" || { echo "Existing TRAIN009C run; refuse overwrite" >&2; exit 3; }
test ! -e "$OUTPUT/run_summary.json" || { echo "Existing completed TRAIN009C run" >&2; exit 3; }
export PYTHONPATH="${QLORA_EXTRA_PYTHONPATH:+$QLORA_EXTRA_PYTHONPATH:}src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" - "$CONFIG" "$DATA_ROOT" "$SFT_DIR" "$RM_DIR" <<'PY'
import json,sys,yaml
from pathlib import Path
from coderl_lab.datasets.train009a_online_prompts import frozen_prompt_pool
from coderl_lab.train.learned_reward_grpo import validate_frozen_models,validate_grpo_contract
config=Path(sys.argv[1]);data=Path(sys.argv[2])
p=Path(sys.argv[3]);r=Path(sys.argv[4])
c=yaml.safe_load(config.read_text())
b=validate_grpo_contract(c,smoke=True)
assert b["optimizer_steps"]==2
assert b["gradient_smoke_allow_truncated"] is False
assert b["natural_eos_required"] is True
assert c["training"]["mask_truncated_completions"] is True
assert c["generation"]["max_completion_length"]==512
assert c["reward_model"]["require_natural_eos"] is True
q=validate_frozen_models(config=c,policy_dir=p,reward_dir=r)
prompts,audit=frozen_prompt_pool(root=data,seed=42,max_train_prompts=16)
assert len(prompts)==16 and audit["eligible_train_prompts"]==4095
print("TRAIN-009C pinned SFT, learned RM, zero RM overlap and natural EOS policy",
 json.dumps({"actor_sha":q["initial_policy_sha256"],
             "RM_sha":q["frozen_reward_sha256"],
             "selected_prompt_sha":audit["selected_prompt_hashes_sha256"]}),flush=True)
PY

(
 flock -n 9 || { echo "GPU occupied; refuse simultaneous model run" >&2; exit 9; }
 "$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 13500
 "$PYTHON" - <<'PY'
import torch,bitsandbytes,trl
from trl import GRPOTrainer
assert torch.cuda.is_available() and torch.cuda.device_count()==1
assert trl.__version__=="1.14.1"
print("TRAIN-009C true EOS GRPO CUDA env",torch.__version__,bitsandbytes.__version__,flush=True)
PY
 "$PYTHON" -m coderl_lab.train.learned_reward_grpo \
   --config "$CONFIG" --data-root "$DATA_ROOT" \
   --sft-dir "$SFT_DIR" --reward-dir "$RM_DIR" \
   --output-dir "$OUTPUT" --smoke --max-train-prompts 16
) 9>/tmp/coderl_lab_gpu_experiment_lock

"$PYTHON" - "$OUTPUT" <<'PY'
import hashlib,json,math,sys
from pathlib import Path
p=Path(sys.argv[1]);s=json.loads((p/"run_summary.json").read_text())
assert s["experiment"]=="TRAIN-009C"
assert s["status"]=="smoke_only" and s["global_step"]==2
assert s["reference_policy"]["actual_pre_update_sft"] is True
assert s["policy_weights_actually_changed"] is True
assert s["at_least_one_real_nonzero_policy_gradient"] is True
assert s["frozen_reward_adapter_unchanged"] is True
assert s["reward_audit"]["rewarded_rollouts"]>=8
assert s["reward_audit"]["natural_eos_terminated"]>=1
assert s["training_identity"]["contract"]["natural_eos_required"] is True
assert any(grad>0 for grad in s["gradient_norm_values"])
assert s["new_policy_adapter_sha256"]==hashlib.sha256(
    (p/"adapter_model.safetensors").read_bytes()).hexdigest()
assert math.isfinite(s["metrics"]["train_loss"])
print("TRAIN-009C real EOS-masked learned-RM GRPO completed (not quality gain)",flush=True)
PY
