#!/usr/bin/env bash
# EXP-004N frozen v5 real-GPU smoke only, NOT efficacy evidence.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
ROOT="artifacts/exp004n"
ADAPTER="artifacts/exp006a/sft-qwen3-1.7b-lora"
DATA="$ROOT/data/public_tasks.jsonl"
SOURCE="ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
test -s "$DATA"
test -x "$PYTHON"
for arm in sft512 base512 sft1024; do
  test ! -e "$ROOT/smoke_$arm/runner.json" || {
    echo "Smoke $arm already frozen; cannot overwrite" >&2; exit 3;
  }
done
test "$(git -C .external/livecodebench rev-parse HEAD)" = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 12000
(
  flock -n 9 || { echo "GPU experiment lock already in use" >&2; exit 9; }
  "$PYTHON" -m coderl_lab.analysis.livecodebench_verifier_gated \
    --experiment-id EXP-004N \
    --public-tasks "$DATA" --model Qwen/Qwen3-1.7B-Base \
    --revision "$SOURCE" --adapter "$ADAPTER" \
    --livecodebench-repo .external/livecodebench \
    --output-dir "$ROOT/smoke_sft512" --max-tasks 2 \
    --max-new-tokens 512 --primary-window 128 \
    --low-threshold 0.05 --high-threshold 0.20 --bias 0.25
  for arm in base512 sft1024; do
    "$PYTHON" -m coderl_lab.analysis.exp004n_capacity_generate \
      --arm "$arm" --public-tasks "$DATA" \
      --manifest "$ROOT/data/manifest.json" \
      --lcb-repo .external/livecodebench \
      --adapter "$ADAPTER" \
      --output-dir "$ROOT/smoke_$arm" --max-tasks 2
  done
) 9>/tmp/coderl_lab_gpu_experiment_lock

"$PYTHON" - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
ids=None
for arm in ("sft512","base512","sft1024"):
  data=json.loads((root/f"smoke_{arm}"/"runner.json").read_text())
  assert data["experiment"]=="EXP-004N"
  assert data["tasks"]==2 and len(data["per_task"])==2
  assert data["private_tests_accessed"] is False
  seen=set(data["per_task"])
  if ids is None: ids=seen
  else: assert seen==ids
print("EXP-004N 2-task THREE-ARM REAL-GPU SMOKE PASS; private tests untouched")
PY
