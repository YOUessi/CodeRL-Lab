#!/usr/bin/env bash
# Optional one-shot Tang execution: DPO smoke only AFTER formal EXP-004N
# has finished with a complete, independently frozen official result.
# This is a host-side command, not a ChatGPT reminder or notification task.
set -euo pipefail
cd "$(dirname "$0")/.."

B_PID="${1:?Usage: queue_posttrain_a_dpo_after_exp004n.sh <B-formal-PID>}"
EXPECTED_A_SHA="${EXPECTED_A_SHA:?Pin the exact Track A Git commit}"
case "$B_PID" in *[!0-9]*|"") echo "B PID must be an integer" >&2; exit 2;; esac
PYTHON="${PYTHON:-.venv/bin/python}"
B_DIR="${B_DIR:-/home/you/projects/CodeRL-Lab/artifacts/exp004n}"
STATE="artifacts/posttrain-a/queue_train007c_after_exp004n_20261009.json"
test -x "$PYTHON" || { echo "Python is missing" >&2; exit 2; }
test ! -e "$STATE" || { echo "Queue already scheduled (no duplicates)" >&2; exit 3; }
mkdir -p "$(dirname "$STATE")"
"$PYTHON" - "$STATE" "$B_PID" "$EXPECTED_A_SHA" <<'PY'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
  "experiment":"TRAIN-007C",
  "purpose":"32/16 NF4 DPO GPU smoke only",
  "stage":"waiting_for_formal_EXP004N",
  "wait_pid":int(sys.argv[2]),
  "expected_A_code_sha":sys.argv[3],
  "automatic_restart":False,
  "new_training_completed":False,
},indent=2)+"\n")
PY
# A single known B experiment process; do not poll other projects or kill jobs.
while kill -0 "$B_PID" 2>/dev/null; do
  state="$(ps -o stat= -p "$B_PID" 2>/dev/null || true)"
  case "$state" in Z*|"") break;; esac
  sleep 30
done

test "$(git rev-parse HEAD)" = "$EXPECTED_A_SHA" || {
  echo "Track A source changed during wait; fail closed" >&2; exit 6;
}
# Do NOT start DPO just because the B process exited. Require official
# held-out v5 completion, after the runner+private freeze barrier.
PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - "$B_DIR" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
paths=[root/p/"runner.json" for p in ("sft512","base512","sft1024")]
for p in paths:
  r=json.loads(p.read_text())
  assert r["experiment"]=="EXP-004N" and r["tasks"]==167
  assert r["private_tests_accessed"] is False
summary=json.loads((root/"final_evaluation/summary.json").read_text())
assert summary["experiment"]=="EXP-004N"
assert summary["tasks"]==167
assert summary["private_tests_accessed_only_after_freeze"] is True
assert summary["bootstrap"]["iterations"]==20000
print("EXP-004N official 167-task result fully frozen and audited",flush=True)
PY

"$PYTHON" - "$STATE" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
data=json.loads(p.read_text());data["stage"]="starting_real_dpo_gpu_smoke"
p.write_text(json.dumps(data,indent=2)+"\n")
PY
# The smoke runner independently obtains GPU lock + validates all SHA pins.
PYTHON="$PYTHON" PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" \
  bash scripts/run_posttrain_a_dpo_smoke.sh > \
  artifacts/posttrain-a/train007c-dpo-smoke-20261009.log 2>&1

test -s artifacts/posttrain-a/train007c-dpo-smoke/run_summary.json
"$PYTHON" - "$STATE" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);x=json.loads(p.read_text())
x["stage"]="dpo_gpu_smoke_completed"
x["new_training_completed"]=True
x["note"]="Only 32 training pairs, 16 heldout, 2 optimizer steps; not formal DPO."
p.write_text(json.dumps(x,indent=2)+"\n")
PY
echo "TRAIN-007C two-step smoke finished on Tang; do not claim 2048-pair formal training."
