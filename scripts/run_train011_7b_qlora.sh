#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv-gpu/bin/python}"
CONFIG="configs/train011_7b_qlora_ultrachat.yaml"
TRAIN="data/generated/posttrain-h4-v1/sft_train.jsonl"
VALID="data/generated/posttrain-h4-v1/sft_validation.jsonl"
MANIFEST="data/generated/posttrain-h4-v1/sft_manifest.json"
PYTHONPATH=src "$PYTHON" - "$TRAIN" "$VALID" "$MANIFEST" <<'PY'
import json,hashlib,sys
from pathlib import Path
tr,va,m=map(Path,sys.argv[1:])
o=json.loads(m.read_text())
assert o["full_scan"] is True and o["stage"]=="sft"
assert o["train"]["rows"]==4096 and o["validation"]["rows"]==256
assert o["prompt_overlap"]==0
for key,path in [("train",tr),("validation",va)]:
    assert hashlib.sha256(path.read_bytes()).hexdigest()==o[key]["sha256"]
print("TRAIN-011 genuine frozen UltraChat data SHA verified",flush=True)
PY
PYTHONPATH=src "$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 11000
if [ "${MODE:-smoke}" = smoke ]; then
  PYTHONPATH=src "$PYTHON" -m coderl_lab.train.sft --config "$CONFIG" --data "$TRAIN" --output-dir artifacts/train011/qlora-ultrachat-smoke-bf16 --max-samples 32 --max-eval-samples 8 --num-train-epochs 1
elif [ "${MODE:-}" = formal ]; then
  PYTHONPATH=src "$PYTHON" -m coderl_lab.train.sft --config "$CONFIG" --data "$TRAIN"
else
  echo "MODE must be smoke or formal" >&2
  exit 2
fi
