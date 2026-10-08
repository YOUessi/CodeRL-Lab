#!/usr/bin/env bash
# Optional on-Tang guarded execution chain for EXP-004N.
# Running this starts a real local process; it is NOT a ChatGPT notification
# or a promise that results have already been produced.
set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN_PID="${1:?Usage: bash scripts/queue_exp004n_after_train.sh <A-training-PID>}"
EXPECTED_SOURCE_SHA="${EXPECTED_SOURCE_SHA:?Set frozen B Git commit SHA}"
case "$TRAIN_PID" in
  *[!0-9]*|"") echo "training PID must be numeric" >&2; exit 2 ;;
esac

ROOT="${ROOT:-artifacts/exp004n}"
mkdir -p "$ROOT"
if [ -e "$ROOT/queue_active.json" ]; then
  echo "Queue already active, refusing second unattended GPU job" >&2
  exit 3
fi

PYTHON="${PYTHON:-.venv/bin/python}"
test -x "$PYTHON" || { echo "missing Python: $PYTHON" >&2; exit 3; }
python3 - "$ROOT/queue_active.json" "$TRAIN_PID" "$EXPECTED_SOURCE_SHA" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "experiment":"EXP-004N", "stage":"waiting_for_A_BF16_GPU_finish",
    "waiting_for_pid":int(sys.argv[2]),
    "expected_code_sha":sys.argv[3],
    "private_tests_accessed":False,
},indent=2)+"\n")
PY
# Wait only for the known TRAIN-007B process, not a guessed wall-clock time.
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  current_state="$(ps -o stat= -p "$TRAIN_PID" 2>/dev/null || true)"
  case "$current_state" in
    Z*|"") break ;;
  esac
  sleep 30
done

# Every condition checked BEFORE allocating new GPU memory.
test "$(git rev-parse HEAD)" = "$EXPECTED_SOURCE_SHA" || {
  echo "B code ref drifted during GPU wait; refusing experiment" >&2; exit 5;
}
test -f "$ROOT/data/manifest.json"
test -f "$ROOT/data/public_tasks.jsonl"
test "$(git -C .external/livecodebench rev-parse HEAD)" = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"

A_DIR="/home/you/projects/CodeRL-Lab-track-a/artifacts/posttrain-a/train007b-lora-ultrachat"
"$PYTHON" - "$A_DIR" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1])
s=json.loads((p/"run_summary.json").read_text())
assert s["num_examples"]==4096 and s["num_heldout_validation_examples"]==256
assert s["quantization"]["mode"]=="none"
assert s["cuda_available"] is True
assert s["saved_adapter_sha256"] == hashlib.sha256((p/"adapter_model.safetensors").read_bytes()).hexdigest()
print("A BF16 full-data training completed and checkpoint verified",flush=True)
PY
PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" \
  -m coderl_lab.train.gpu_preflight --min-free-mib 12000

# Avoid two CodeRL-Lab launchers allocating the same physical GPU together.
(
  flock -n 9 || { echo "CodeRL-Lab GPU lock occupied" >&2; exit 8; }
  if [ -e "$ROOT/sft512/runner.json" ] || [ -e "$ROOT/base512/runner.json" ] || [ -e "$ROOT/sft1024/runner.json" ]; then
    echo "One or more output runners already frozen, refusing overwrite" >&2
    exit 9
  fi
  bash scripts/run_exp004n_generate.sh > "$ROOT/official_generate.log" 2>&1
  test -s "$ROOT/sft512/runner.json"
  test -s "$ROOT/base512/runner.json"
  test -s "$ROOT/sft1024/runner.json"
  # No evaluation before all model decisions have been frozen.
  bash scripts/eval_exp004n.sh > "$ROOT/official_private_eval.log" 2>&1
) 9>/tmp/coderl_lab_gpu_experiment_lock

test -s "$ROOT/final_evaluation/summary.json"
"$PYTHON" - "$ROOT/queue_active.json" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
x=json.loads(p.read_text())
x["stage"]="official_private_evaluation_completed"
p.write_text(json.dumps(x,indent=2)+"\n")
print("EXP-004N finished on-device. Summaries require separate review and GitHub archival.")
PY
