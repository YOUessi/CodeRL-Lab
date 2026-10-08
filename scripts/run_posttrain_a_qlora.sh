#!/usr/bin/env bash
# Real GPU training is intentionally NOT auto-launched from CI.
# EXP-004L is using the same Tang GPU; only run after it has exited.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

DATA="data/generated/posttrain-h4-v1/sft_train.jsonl"
EVAL="data/generated/posttrain-h4-v1/sft_validation.jsonl"
MANIFEST="data/generated/posttrain-h4-v1/sft_manifest.json"
CONFIG="configs/posttrain_a_qlora_qwen3_1.7b.yaml"

python - "$DATA" "$EVAL" "$MANIFEST" "$CONFIG" <<'PY'
import hashlib, json, pathlib, sys
import yaml
train_path, eval_path, manifest_path, config_path = map(pathlib.Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
assert manifest["track"] == "A-posttraining"
assert manifest["stage"] == "sft"
assert manifest["full_scan"] is True, "SMOKE DATA MUST NEVER ENTER FORMAL TRAINING"
assert manifest["test_or_private_examples_in_training"] is False
assert manifest["dataset_revision"] == config["data"]["revision"]
assert manifest["train"]["rows"] == config["data"]["expected_train_rows"]
assert manifest["validation"]["rows"] == config["data"]["expected_validation_rows"]
for key, path in (("train", train_path), ("validation", eval_path)):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[key]["sha256"]
    assert sum(1 for line in path.open(encoding="utf-8") if line.strip()) == manifest[key]["rows"]
assert manifest["prompt_overlap"] == 0
print("Track A train/validation manifest, versions, sizes, disjointness and SHA256 OK")
PY

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU training requires nvidia-smi" >&2
  exit 3
fi
# No parallel runs on Tang: benchmark fidelity and OOM risk matter.
GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -E '^[0-9]+' || true)"
if [ -n "$GPU_PIDS" ]; then
  echo "Tang GPU is already busy; refusing concurrent QLoRA training:" >&2
  echo "$GPU_PIDS" >&2
  exit 4
fi

python - <<'PY'
import torch
import bitsandbytes
if not torch.cuda.is_available():
    raise RuntimeError("QLoRA requires a real CUDA GPU")
print("CUDA:", torch.cuda.get_device_name(0))
print("bitsandbytes:", getattr(bitsandbytes, "__version__", "unknown"))
PY

python -m coderl_lab.train.sft --config "$CONFIG" --data "$DATA"
