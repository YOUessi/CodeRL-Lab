#!/usr/bin/env bash
# TRAIN-010A tests actual Torch autograd PPO on CUDA, not trained Qwen PPO.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
test -x "$PYTHON" || { echo "Pinned Torch venv unavailable" >&2; exit 2; }
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
(
  flock -n 9 || { echo "GPU busy; do not interrupt other model experiment" >&2; exit 9; }
  "$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 13500
  "$PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count()==1
print("TRAIN-010A verifying actual GPU gradients",torch.__version__,torch.cuda.get_device_name(0),flush=True)
PY
  "$PYTHON" -m pytest -q -v \
     tests/test_ppo_objectives.py tests/test_ppo_torch_objectives.py
) 9>/tmp/coderl_lab_gpu_experiment_lock
echo "TRAIN-010A CPU + REAL CUDA PPO objective numerical/gradient tests passed. NOT full PPO trainer."
