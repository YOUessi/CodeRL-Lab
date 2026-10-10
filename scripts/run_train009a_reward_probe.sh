#!/usr/bin/env bash
# TRAIN-009A: frozen four-prompt greedy policy/Reward-Model proxy probe.
# Post-training only; no optimizer updates and no human-preference claim.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/you/projects/CodeRL-Lab-track-a/data/generated/posttrain-h4-v1}"
SFT_DIR="${SFT_DIR:-/home/you/projects/CodeRL-Lab-track-a/artifacts/posttrain-a/train007a-qlora-ultrachat}"
RM_DIR="${RM_DIR:-/home/you/projects/CodeRL-Lab-track-rm/artifacts/posttrain-a/train008a-reward-model}"
ROOT="artifacts/posttrain-a/train009a-ppo-clip-smoke"
OUTPUT="$ROOT/probe_summary.json"
export PYTHONPATH="${QLORA_EXTRA_PYTHONPATH:+$QLORA_EXTRA_PYTHONPATH:}src${PYTHONPATH:+:$PYTHONPATH}"

test -x "$PYTHON" || { echo "Python missing" >&2; exit 2; }
test -s "$ROOT/run_summary.json" || { echo "PPO 4step model not complete" >&2; exit 3; }
test ! -e "$OUTPUT" || { echo "PPO probe result exists, refusing overwrite" >&2; exit 4; }

(
  flock -n 9 || { echo "CodeRL GPU mutex occupied" >&2; exit 9; }
  "$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 11000
  "$PYTHON" - <<'PY'
import torch,bitsandbytes
assert torch.cuda.is_available()
print("TRAIN-009A probe CUDA/BnB verified",bitsandbytes.__version__,flush=True)
PY
  "$PYTHON" -m coderl_lab.analysis.train009a_probe \
    --config configs/train009a_ppo_clip_rlhf_smoke.yaml \
    --data-root "$DATA_ROOT" \
    --sft-adapter "$SFT_DIR" \
    --reward-adapter "$RM_DIR" \
    --ppo-root "$ROOT" \
    --output "$OUTPUT"
) 9>/tmp/coderl_lab_gpu_experiment_lock

"$PYTHON" - "$OUTPUT" <<'PY'
import json,math,sys
from pathlib import Path
p=Path(sys.argv[1])
r=json.loads(p.read_text())
assert r["experiment"]=="TRAIN-009A-probe"
assert r["probe_count"]==4
assert r["no_human_labels_used"] is True
assert r["no_optimizer_updates"] is True
assert len(r["probe_rows"])==4
assert math.isfinite(r["mean_reward_proxy_score_difference"])
assert len(set(row["prompt_sha256"] for row in r["probe_rows"]))==4
print("TRAIN-009A true GPU frozen four-probe reward diagnostics PASS",flush=True)
PY
