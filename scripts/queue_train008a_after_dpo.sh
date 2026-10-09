#!/usr/bin/env bash
# User-authorized single-use Tang serial GPU queue: RM smoke only if formal DPO
# finishes and its frozen 2048/256 model artifacts pass integrity checks.
# No reminder, cloud background orchestration or user notification is implied.
set -euo pipefail
cd "$(dirname "$0")/.."

DPO_PID="${1:?Pass exact running TRAIN-007C parent PID}"
EXPECTED_RM_SHA="${EXPECTED_RM_SHA:?Pin the exact already-tested RM Git source SHA}"
case "$DPO_PID" in *[!0-9]*|"") echo "DPO PID must be numeric" >&2; exit 2;; esac

PYTHON="${PYTHON:-.venv/bin/python}"
DPO_DIR="${DPO_DIR:-/home/you/projects/CodeRL-Lab-track-a/artifacts/posttrain-a/train007c-dpo-ultrafeedback}"
ROOT="artifacts/posttrain-a"
STATE="$ROOT/queue_train008a_after_dpo.json"
LOG="$ROOT/train008a-reward-model-smoke-20261009.log"
test -x "$PYTHON" || { echo "Python missing" >&2; exit 2; }
test "$(git rev-parse HEAD)" = "$EXPECTED_RM_SHA" || {
  echo "RM source drift before queue" >&2; exit 3;
}
test ! -e "$STATE" && test ! -e "$LOG" || {
  echo "Queue previously created; refusing duplicate" >&2; exit 4;
}
mkdir -p "$ROOT"
"$PYTHON" - "$STATE" "$DPO_PID" "$EXPECTED_RM_SHA" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "experiment":"TRAIN-008A",
    "mode":"32x16_real_GPU_smoke",
    "stage":"waiting_for_frozen_TRAIN007C_result",
    "wait_pid":int(sys.argv[2]),
    "expected_RM_git_sha":sys.argv[3],
    "gpu_training_completed":False,
},indent=2)+"\n")
PY

while kill -0 "$DPO_PID" 2>/dev/null; do
  status="$(ps -o stat= -p "$DPO_PID" 2>/dev/null || true)"
  case "$status" in Z*|"") break;; esac
  sleep 30
done
test "$(git rev-parse HEAD)" = "$EXPECTED_RM_SHA" || {
  echo "RM source changed during wait; fail closed" >&2; exit 5;
}

# Never start a second model simply because DPO process exited. Ensure it
# finished successfully and saved a SHA-matched adapter and heldout metrics.
"$PYTHON" - "$DPO_DIR" <<'PY'
from pathlib import Path
import hashlib,json,math,sys
root=Path(sys.argv[1])
s=json.loads((root/"run_summary.json").read_text())
h=json.loads((root/"log_history.json").read_text())
weight=root/"adapter_model.safetensors"
assert s["num_pairs"]==2048
assert s["num_heldout_preference_pairs"]==256
assert s["global_step"]>200
assert s["quantization"]["mode"]=="nf4"
assert s["sft_adapter_sha256"]=="d8846aa5fb5fd6958d30a24611efd9fb99eb91469a60192937646b58557ce12d"
assert s["checkpoint_policy"]["identity"]["training_data_sha256"]=="945a336244462d43a7c3e70774158a4c40e873b5ee4f6c93d73e0d1e82222b77"
assert s["checkpoint_policy"]["identity"]["heldout_data_sha256"]=="9d8a62dee49e9841add47a2ed27e485926c41e988ac06185ccd23b0df4e2e450"
assert s["output_adapter_sha256"]==hashlib.sha256(weight.read_bytes()).hexdigest()
eval_rows=[x for x in h if "eval_loss" in x]
assert eval_rows and math.isfinite(eval_rows[-1]["eval_loss"])
print("Frozen full DPO completed; starting independent reward-model smoke only now",flush=True)
PY

# The model launch itself performs GPU exclusive and artifact guards.
"$PYTHON" - "$STATE" <<'PY'
from pathlib import Path
import json,sys
p=Path(sys.argv[1]);s=json.loads(p.read_text())
s["stage"]="starting_NF4_reward_smoke"
p.write_text(json.dumps(s,indent=2)+"\n")
PY
test ! -e "$ROOT/train008a-reward-model-smoke/training_identity.json" || {
  echo "Reward smoke already started; cannot overwrite" >&2; exit 6;
}
PYTHON="$PYTHON" MODE=smoke QLORA_EXTRA_PYTHONPATH="${QLORA_EXTRA_PYTHONPATH:-}" \
  bash scripts/run_posttrain_a_reward_model.sh > "$LOG" 2>&1

"$PYTHON" - "$STATE" "$ROOT/train008a-reward-model-smoke/run_summary.json" <<'PY'
from pathlib import Path
import json,sys
p=Path(sys.argv[1]);sm=json.loads(Path(sys.argv[2]).read_text())
assert sm["experiment"]=="TRAIN-008A"
assert sm["train_pairs"]==32 and sm["heldout_pairs"]==16
assert sm["optimizer_steps"]==4
assert sm["status"]=="smoke_only"
s=json.loads(p.read_text());s["stage"]="reward_model_smoke_completed"
s["gpu_training_completed"]=True
p.write_text(json.dumps(s,indent=2)+"\n")
print("Reward-model smoke finished; NOT formal 2048/256 reward training.")
PY
