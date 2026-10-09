#!/usr/bin/env bash
# Single-use Tang host queue (no ChatGPT async notification): run TRAIN-007D
# only after full TRAIN-008A reward GPU training has FINISHED and passed SHA.
set -euo pipefail
cd "$(dirname "$0")/.."
RM_PARENT_PID="${1:?Usage: queue_train007d_after_rm.sh <confirmed-reward-training-PID>}"
EXPECTED_EVAL_GIT_SHA="${EXPECTED_EVAL_GIT_SHA:?Pin frozen eval git SHA}"
case "$RM_PARENT_PID" in *[!0-9]*|"") echo "invalid reward training PID" >&2; exit 2 ;; esac
PYTHON="${PYTHON:-.venv/bin/python}"
RM_DIR="${RM_DIR:-/home/you/projects/CodeRL-Lab-track-rm/artifacts/posttrain-a/train008a-reward-model}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/train007d-matched-preference}"
STATE="$OUTPUT_DIR/queue_state.json"
LOG="$OUTPUT_DIR/paired_sft_vs_dpo_gpu_20261010.log"
test -x "$PYTHON" || { echo "Python missing" >&2; exit 2; }
test ! -e "$STATE" && test ! -e "$LOG" || { echo "queue already started" >&2; exit 5; }
test "$(git rev-parse HEAD)" = "$EXPECTED_EVAL_GIT_SHA" || { echo "unfrozen source" >&2; exit 5; }
mkdir -p "$OUTPUT_DIR"
"$PYTHON" - "$STATE" "$RM_PARENT_PID" "$EXPECTED_EVAL_GIT_SHA" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
 "experiment":"TRAIN-007D",
 "stage":"waiting_for_verified_full_reward_gpu",
 "dependency_pid":int(sys.argv[2]),
 "pinned_source_sha":sys.argv[3],
 "model_evaluation_completed":False,
},indent=2)+"\n")
PY
while kill -0 "$RM_PARENT_PID" 2>/dev/null; do
  code="$(ps -o stat= -p "$RM_PARENT_PID" 2>/dev/null || true)"
  case "$code" in Z*|"") break;; esac
  sleep 30
done
test "$(git rev-parse HEAD)" = "$EXPECTED_EVAL_GIT_SHA" || {
 echo "evaluation code drifted after waiting; refusing GPU" >&2; exit 5;
}

# Training process exit alone is NOT proof of success. No GPU reservation unless
# complete 2048/256 reward-model checkpoint and validation summary are valid.
"$PYTHON" - "$RM_DIR" <<'PY'
from pathlib import Path
import json,hashlib,math,sys
root=Path(sys.argv[1])
s=json.loads((root/"run_summary.json").read_text())
w=root/"adapter_model.safetensors"
assert s["experiment"]=="TRAIN-008A" and s["status"]=="formal"
assert s["train_pairs"]==2048 and s["heldout_pairs"]==256
assert s["optimizer_steps"]==s["total_steps_expected"]==256
assert s["training_identity"]["formal"] is True
assert s["training_identity"]["train_file_sha256"]=="945a336244462d43a7c3e70774158a4c40e873b5ee4f6c93d73e0d1e82222b77"
assert s["training_identity"]["heldout_file_sha256"]=="9d8a62dee49e9841add47a2ed27e485926c41e988ac06185ccd23b0df4e2e450"
assert s["adapter_sha256"]==hashlib.sha256(w.read_bytes()).hexdigest()
assert s["initial_heldout"]["pairs"]==s["heldout"]["pairs"]==256
assert math.isfinite(s["heldout"]["mean_pairwise_loss"])
print("Full TRAIN-008A reward adapter and 256 heldout verified; subsequent eval is independent SFT/DPO.",flush=True)
PY

# Exact separate artifact roots; fail closed if already producing a result.
"$PYTHON" - "$STATE" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);x=json.loads(p.read_text());x["stage"]="evaluating_same_252_heldout_pairs"
p.write_text(json.dumps(x,indent=2)+"\n")
PY

PYTHON="$PYTHON" \
QLORA_EXTRA_PYTHONPATH="${QLORA_EXTRA_PYTHONPATH:-}" \
ROOT="$OUTPUT_DIR" \
bash scripts/run_train007d_matched_eval.sh > "$LOG" 2>&1

"$PYTHON" - "$STATE" "$OUTPUT_DIR/summary.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);r=json.loads(Path(sys.argv[2]).read_text())
assert r["experiment"]=="TRAIN-007D"
assert r["effective_validation_pairs"]==252
assert r["frozen_inputs"]["no_private_or_hidden_tests"] is True
q=json.loads(p.read_text())
q["stage"]="matched_252_pair_GPU_evaluation_completed"
q["model_evaluation_completed"]=True
p.write_text(json.dumps(q,indent=2)+"\n")
print("TRAIN-007D frozen 252-pair GPU evaluation finished.")
PY
