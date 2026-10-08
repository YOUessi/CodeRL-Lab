#!/usr/bin/env bash
# GPU smoke on 32 REAL held-out-clean UltraChat samples; do not run beside B.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
TRAIN="data/generated/posttrain-h4-v1/sft_train.jsonl"
VALID="data/generated/posttrain-h4-v1/sft_validation.jsonl"
MANIFEST="data/generated/posttrain-h4-v1/sft_manifest.json"
CONFIG="configs/posttrain_a_qlora_qwen3_1.7b.yaml"
OUTPUT="${OUTPUT:-artifacts/posttrain-a/train007a-qlora-smoke}"

test -x "$PYTHON" || { echo "Python interpreter unavailable: $PYTHON" >&2; exit 2; }
PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" - "$TRAIN" "$VALID" "$MANIFEST" "$CONFIG" <<'PY'
import hashlib, json, sys
from pathlib import Path
import yaml
train, valid, manifest, config = map(Path, sys.argv[1:])
m = json.loads(manifest.read_text())
cfg = yaml.safe_load(config.read_text())
assert m["stage"] == "sft" and m["full_scan"] is True
assert m["train"]["rows"] == 4096 and m["validation"]["rows"] == 256
assert m["dataset_revision"] == cfg["data"]["revision"]
assert hashlib.sha256(train.read_bytes()).hexdigest() == m["train"]["sha256"]
assert hashlib.sha256(valid.read_bytes()).hexdigest() == m["validation"]["sha256"]
assert m["prompt_overlap"] == 0
print("TRAIN-007A real dataset checksum and split isolation OK")
PY

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "CUDA GPU with nvidia-smi required" >&2
  exit 3
fi
PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 6000

PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON" -m coderl_lab.train.sft \
  --config "$CONFIG" \
  --data "$TRAIN" \
  --output-dir "$OUTPUT" \
  --max-samples 32 \
  --max-eval-samples 8 \
  --num-train-epochs 1

echo "Smoke artifacts: $OUTPUT"
