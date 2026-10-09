#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

ROOT="${ROOT:-artifacts/exp004l/smoke}"
PUBLIC_TASKS="${PUBLIC_TASKS:-artifacts/exp004l/data/public_tasks.jsonl}"
MANIFEST="${MANIFEST:-artifacts/exp004l/data/manifest.json}"
LCB_ROOT="${LCB_ROOT:-.external/livecodebench}"
ADAPTER="${ADAPTER:-artifacts/exp006a/sft-qwen3-1.7b-lora}"
PINNED_LCB="28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
PINNED_DATA="0fe84c3912ea0c4d4a78037083943e8f0c4dd505"

python - "$MANIFEST" "$PUBLIC_TASKS" <<'PY'
import hashlib, json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
public = pathlib.Path(sys.argv[2])
assert manifest["source_revision"] == "0fe84c3912ea0c4d4a78037083943e8f0c4dd505", manifest
assert manifest["tasks"] == 175, manifest
assert hashlib.sha256(public.read_bytes()).hexdigest() == manifest["public_view_sha256"]
assert manifest["private_tests_exported"] is False
assert manifest["private_tests_decoded"] is False
print("EXP-004L: dataset version, counts, public view and no-private manifest verified")
PY

test "$(git -C "$LCB_ROOT" rev-parse HEAD)" = "$PINNED_LCB"
test -d "$ADAPTER"

mkdir -p "$ROOT"
python -m coderl_lab.datasets.livecodebench_smoke \
  --public-tasks "$PUBLIC_TASKS" \
  --output "$ROOT/public_tasks_12.jsonl"

python -m coderl_lab.analysis.livecodebench_verifier_gated \
  --public-tasks "$ROOT/public_tasks_12.jsonl" \
  --model "Qwen/Qwen3-1.7B-Base" \
  --revision "ea980cb0a6c2ae4b936e82123acc929f1cec04c1" \
  --adapter "$ADAPTER" \
  --livecodebench-repo "$LCB_ROOT" \
  --output-dir "$ROOT" \
  --max-new-tokens 512 \
  --primary-window 128 \
  --low-threshold 0.05 \
  --high-threshold 0.20 \
  --bias 0.25

python - "$ROOT/runner.json" <<'PY'
import json, pathlib, sys
runner = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert runner["tasks"] == 12
assert runner["private_tests_accessed"] is False
assert len(runner["per_task"]) == 12
assert all(
    x["gated_low_destabilize"]["raw_completion"] == x["baseline"]["raw_completion"]
    and x["gated_high_destabilize"]["raw_completion"] == x["baseline"]["raw_completion"]
    for x in runner["per_task"].values() if x["public_all_pass"]
)
print("EXP-004L smoke: 12 model tasks complete, no private leakage, public-pass protection OK")
PY

echo "Smoke generation complete: $ROOT/runner.json"
echo "Hidden evaluation is deliberately a separate, post-freeze phase."
