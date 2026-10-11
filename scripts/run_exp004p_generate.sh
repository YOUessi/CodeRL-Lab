#!/usr/bin/env bash
# Historical v4 independent cohort; exploratory, not prospective newer data.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv-gpu/bin/python}"
ROOT="${ROOT:-artifacts/exp004p}"
ADAPTER="${ADAPTER:-artifacts/exp006a/sft-qwen3-1.7b-lora}"
LCB="${LCB_ROOT:-.external/livecodebench}"
test -x "$PYTHON"
test "$(git -C "$LCB" rev-parse HEAD)" = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
test -s "$ADAPTER/adapter_model.safetensors"
mkdir -p "$ROOT/data"
PYTHONPATH=src "$PYTHON" -m coderl_lab.datasets.exp004p_public_v4 --output-dir "$ROOT/data"
PYTHONPATH=src "$PYTHON" - "$ROOT/data" "$ADAPTER" <<'PY'
import json,sys
from pathlib import Path
from coderl_lab.datasets.livecodebench import sha256_file
d,a=map(Path,sys.argv[1:])
m=json.loads((d/'manifest.json').read_text())
assert m['accepted_count']==101
assert m['public_sha256']=='62239669ec50b2ebee8cad1a0561e072e293a76a9a130d0b214bdbcedf980853'
assert m['sources_sha256']['test4.jsonl']=='d711138ddaebfcf5f8ec6a4283ee677298c0f5c5d374a235af92aaf0584510da'
assert sha256_file(d/'public_tasks.jsonl')==m['public_sha256']
assert sha256_file(a/'adapter_model.safetensors')=='e1ea9a007a3e4213cdeb1ca6b1e12d29adb83594991421757d0d40b9168d0861'
assert m['private_tests_decoded'] is False
print("101-task public-only, frozen SFT adapter verified",flush=True)
PY
test ! -f "$ROOT/sft512/runner.json" || { echo 'runner exists: refusing overwrite' >&2; exit 5; }
PYTHONPATH=src "$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 12000
N=()
if [ "${SMOKE_TASKS:-0}" -gt 0 ]; then N=(--max-tasks "$SMOKE_TASKS"); fi
PYTHONPATH=src "$PYTHON" -m coderl_lab.analysis.livecodebench_verifier_gated \
 --experiment-id EXP-004P --public-tasks "$ROOT/data/public_tasks.jsonl" \
 --model Qwen/Qwen3-1.7B-Base \
 --revision ea980cb0a6c2ae4b936e82123acc929f1cec04c1 \
 --adapter "$ADAPTER" --livecodebench-repo "$LCB" \
 --output-dir "$ROOT/sft512" --max-new-tokens 512 \
 --primary-window 128 --low-threshold 0.05 --high-threshold 0.20 --bias 0.25 "${N[@]}"
