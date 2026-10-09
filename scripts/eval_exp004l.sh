#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

ROOT="${ROOT:-artifacts/exp004l/full}"
EXPECTED_TASKS="${EXPECTED_TASKS:-175}"
MANIFEST="artifacts/exp004l/data/manifest.json"
PUBLIC_TASKS="artifacts/exp004l/data/public_tasks.jsonl"

test -f "$ROOT/runner.json" || { echo "No frozen runner: $ROOT/runner.json" >&2; exit 2; }

python -m coderl_lab.analysis.livecodebench_formal_eval \
  --runner "$ROOT/runner.json" \
  --public-tasks "$PUBLIC_TASKS" \
  --manifest "$MANIFEST" \
  --lcb-repo ".external/livecodebench" \
  --output-dir "$ROOT/final_evaluation" \
  --expected-tasks "$EXPECTED_TASKS" \
  --iterations 20000 \
  --seed 42 \
  --timeout 6 \
  --memory 1g

echo "Official LiveCodeBench results: $ROOT/final_evaluation/summary.json"
