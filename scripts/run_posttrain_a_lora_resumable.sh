#!/usr/bin/env bash
# TRAIN-007B replacement for interrupted 146-step no-checkpoint run.
# Restart from zero; later interruptions can resume from exact optimizer/RNG step.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
CONFIG="configs/posttrain_a_lora_resumable_qwen3_1.7b.yaml"
DATA_ROOT="data/generated/posttrain-h4-v1"
OUTPUT="artifacts/posttrain-a/train007b-lora-resumable"
ARGS=()
if [ -n "${RESUME_CHECKPOINT:-}" ]; then
  ARGS+=(--resume-from-checkpoint "$RESUME_CHECKPOINT")
fi
test -x "$PYTHON" || { echo "missing Python: $PYTHON" >&2; exit 2; }
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" - "$CONFIG" "$DATA_ROOT" <<'PY'
import hashlib, json, sys
from pathlib import Path
import yaml
cfg= yaml.safe_load(Path(sys.argv[1]).read_text())
root=Path(sys.argv[2])
m=json.loads((root/"sft_manifest.json").read_text())
assert m["stage"]=="sft" and m["full_scan"] is True and m["prompt_overlap"]==0
assert (m["train"]["rows"],m["validation"]["rows"])==(4096,256)
assert m["dataset_revision"]==cfg["data"]["revision"]
assert cfg["training"]["save_strategy"]=="steps"
assert cfg["training"]["save_steps"]==32
for k in ("train","validation"):
 p=root/f"sft_{k}.jsonl"
 assert hashlib.sha256(p.read_bytes()).hexdigest()==m[k]["sha256"]
print("Frozen 4096/256 UltraChat data and step-checkpoint policy verified")
PY
"$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 12000

# This lock is shared with the independent B GPU launcher.
(
  flock -n 9 || { echo "CodeRL-Lab GPU experiment lock held" >&2; exit 9; }
  "$PYTHON" -m coderl_lab.train.sft \
    --config "$CONFIG" --data "$DATA_ROOT/sft_train.jsonl" \
    --output-dir "$OUTPUT" "${ARGS[@]}"
) 9>/tmp/coderl_lab_gpu_experiment_lock
