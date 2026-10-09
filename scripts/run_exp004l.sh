#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

ROOT="${ROOT:-artifacts/exp004l/full}"
PUBLIC_TASKS="artifacts/exp004l/data/public_tasks.jsonl"
MANIFEST="artifacts/exp004l/data/manifest.json"
ADAPTER="${ADAPTER:-artifacts/exp006a/sft-qwen3-1.7b-lora}"
LCB_ROOT="${LCB_ROOT:-.external/livecodebench}"

python - "$MANIFEST" "$PUBLIC_TASKS" <<'PY'
from coderl_lab.datasets.livecodebench import sha256_file
from pathlib import Path
import json, sys
m = json.loads(Path(sys.argv[1]).read_text())
assert m["tasks"] == 175
assert m["source_revision"] == "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
assert m["public_view_sha256"] == sha256_file(Path(sys.argv[2]))
assert m["private_tests_exported"] is False
assert m["private_tests_decoded"] is False
print("175-task pinned dataset and public-only view validated")
PY
test "$(git -C "$LCB_ROOT" rev-parse HEAD)" = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
test -d "$ADAPTER"
test ! -f "$ROOT/runner.json" || {
  echo "Formal runner already exists: $ROOT/runner.json; refusing to overwrite frozen results" >&2
  exit 3
}

mkdir -p "$ROOT"
python -m coderl_lab.analysis.livecodebench_verifier_gated \
  --public-tasks "$PUBLIC_TASKS" \
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
import json, sys
from pathlib import Path
r = json.loads(Path(sys.argv[1]).read_text())
assert r["tasks"] == len(r["per_task"]) == 175
assert r["private_tests_accessed"] is False
assert all(
    (x["gate_triggered"] == (not x["public_all_pass"] and x["eligible"]))
    for x in r["per_task"].values()
)
assert all(
    x["gated_low_destabilize"]["raw_completion"] == x["baseline"]["raw_completion"]
    and x["gated_high_destabilize"]["raw_completion"] == x["baseline"]["raw_completion"]
    for x in r["per_task"].values() if x["public_all_pass"]
)
print("175/175 generated and frozen; public-gate invariant passed; private tests untouched")
PY
echo "Only now run: ROOT=$ROOT EXPECTED_TASKS=175 bash scripts/eval_exp004l.sh"
