#!/usr/bin/env bash
# EXP-004N: full v5 (167 tasks), never selects tasks based on hidden outcomes.
# GPU-intensive; run only when Track A no longer occupies Tang.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
ROOT="${ROOT:-artifacts/exp004n}"
DATA="$ROOT/data"
LCB="${LCB_ROOT:-.external/livecodebench}"
ADAPTER="${ADAPTER:-artifacts/exp006a/sft-qwen3-1.7b-lora}"

test -x "$PYTHON" || { echo "Python unavailable: $PYTHON" >&2; exit 2; }
test "$(git -C "$LCB" rev-parse HEAD)" = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
test -s "$ADAPTER/adapter_model.safetensors"

if [ ! -s "$DATA/manifest.json" ]; then
  PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" \
    -m coderl_lab.datasets.livecodebench_v5 --output-dir "$DATA"
fi

PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - "$DATA" "$ADAPTER" <<'PY'
import json, sys
from pathlib import Path
from coderl_lab.datasets.livecodebench import sha256_file
from coderl_lab.analysis.exp004n_capacity_generate import (
    SOURCE_SHA256, PUBLIC_SHA256, SOURCE_REVISION, SFT_ADAPTER_SHA256,
)
root, adapter = map(Path, sys.argv[1:])
manifest = json.loads((root/"manifest.json").read_text())
assert manifest["fine_grained_version"]=="v5"
assert manifest["source_revision"]==SOURCE_REVISION
assert manifest["source_sha256"]==SOURCE_SHA256
assert manifest["public_view_sha256"]==sha256_file(root/"public_tasks.jsonl")==PUBLIC_SHA256
assert manifest["tasks"]==167
assert manifest["private_tests_exported"] is False
assert manifest["private_tests_decoded"] is False
assert sha256_file(adapter/"adapter_model.safetensors")==SFT_ADAPTER_SHA256
print("EXP-004N frozen v5 167-task public-only and Qwen3 adapter audit PASSED")
PY

for arm in sft512 base512 sft1024; do
  if [ -e "$ROOT/$arm/runner.json" ]; then
    echo "Frozen $arm already exists; refusing implicit overwrite or selective rerun" >&2
    exit 5
  fi
done
PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" \
  -m coderl_lab.train.gpu_preflight --min-free-mib 12000

# Fixed order: original SFT decoding mechanism first, then Base and length controls.
PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" \
  -m coderl_lab.analysis.livecodebench_verifier_gated \
  --experiment-id EXP-004N --public-tasks "$DATA/public_tasks.jsonl" \
  --model Qwen/Qwen3-1.7B-Base \
  --revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 \
  --adapter "$ADAPTER" --livecodebench-repo "$LCB" \
  --output-dir "$ROOT/sft512" \
  --max-new-tokens 512 --primary-window 128 \
  --low-threshold 0.05 --high-threshold 0.20 --bias 0.25

for arm in base512 sft1024; do
  PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" \
    -m coderl_lab.analysis.exp004n_capacity_generate \
    --arm "$arm" --public-tasks "$DATA/public_tasks.jsonl" \
    --manifest "$DATA/manifest.json" --lcb-repo "$LCB" \
    --adapter "$ADAPTER" --output-dir "$ROOT/$arm"
done

PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
for arm in ("sft512","base512","sft1024"):
    r=json.loads((p/arm/"runner.json").read_text())
    assert r["experiment"]=="EXP-004N",arm
    assert r["tasks"]==167,len(r["per_task"])==167
    assert r["private_tests_accessed"] is False
print("Frozen EXP-004N: all 167 x 3 generated, private cases untouched")
PY
echo "Generation completed. Only now run scripts/eval_exp004n.sh for private cases."
