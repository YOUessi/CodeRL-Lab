#!/usr/bin/env bash
# Smoke exercises *real* frozen HF SFT rows, but scans only first 64 source
# examples, explicitly forbidding interpretation as TRAIN-007A formal training.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
test -x "$PYTHON" || { echo "Python interpreter missing: $PYTHON" >&2; exit 2; }

# Optional overlay for bitsandbytes installed outside the active venv.
export PYTHONPATH="${QLORA_EXTRA_PYTHONPATH:-}:src${PYTHONPATH:+:$PYTHONPATH}"
DATA_ROOT="data/generated/posttrain-h4-smoke"
OUTPUT="${OUTPUT:-artifacts/posttrain-a/train007a-qlora-hf-smoke}"
CONFIG="configs/posttrain_a_qlora_hf_smoke_qwen3_1.7b.yaml"

if [ ! -s "$DATA_ROOT/sft_manifest.json" ]; then
  "$PYTHON" -m coderl_lab.datasets.posttrain_h4 \
    --stage sft --output-dir "$DATA_ROOT" \
    --train-size 32 --validation-size 8 --smoke-scan-limit 64
fi

"$PYTHON" - "$DATA_ROOT" <<'PY'
import hashlib, json, sys
from pathlib import Path
p = Path(sys.argv[1])
m = json.loads((p / "sft_manifest.json").read_text())
assert m["full_scan"] is False and m["scan_limit"] == 64
assert m["stage"] == "sft" and m["prompt_overlap"] == 0
assert m["train"]["rows"] == 32 and m["validation"]["rows"] == 8
for side in ("train", "validation"):
    f = p / f"sft_{side}.jsonl"
    assert hashlib.sha256(f.read_bytes()).hexdigest() == m[side]["sha256"]
print("Verified 32/8 REAL HF SMOKE samples (NOT formal full-stream training)")
PY

"$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 6000

if [ -e "$OUTPUT/adapter_model.safetensors" ]; then
  echo "Smoke adapter already exists: refusing overwrite" >&2
  exit 3
fi

"$PYTHON" -m coderl_lab.train.sft \
  --config "$CONFIG" --data "$DATA_ROOT/sft_train.jsonl" \
  --output-dir "$OUTPUT" --max-samples 32 --max-eval-samples 8

"$PYTHON" - "$OUTPUT/run_summary.json" "$OUTPUT/adapter_model.safetensors" <<'PY'
import hashlib, json, math, sys
from pathlib import Path
s = json.loads(Path(sys.argv[1]).read_text())
weights = Path(sys.argv[2])
assert s["num_examples"] == 32
assert s["num_heldout_validation_examples"] == 8
assert s["quantization"]["mode"] == "nf4"
assert s["cuda_available"] is True
assert s["saved_adapter_sha256"] == hashlib.sha256(weights.read_bytes()).hexdigest()
assert math.isfinite(s["metrics"]["train_loss"])
assert s["metrics"]["train_loss"] >= 0
assert len(json.loads(Path(sys.argv[1]).with_name("log_history.json").read_text())) > 0
print("NF4 real-HF SFT smoke PASS, loss =", s["metrics"]["train_loss"])
PY
