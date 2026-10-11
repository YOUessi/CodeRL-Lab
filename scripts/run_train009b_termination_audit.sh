#!/usr/bin/env bash
# TRAIN-009B: frozen, no-weight-update SFT EOS diagnostic.
# One 512-token sampled trace per 16 train + 16 heldout prompts; derive
# prefix budgets 96/256/512 without regenerated/cherry-picked completions.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
A_WORKTREE="${A_WORKTREE:-/home/you/projects/CodeRL-Lab-track-a}"
ROOT="artifacts/train009b-termination-audit"
SFT="$A_WORKTREE/artifacts/posttrain-a/train007a-qlora-ultrachat"
DATA="$A_WORKTREE/data/generated/posttrain-h4-v1"
test -x "$PYTHON" || { echo "Python interpreter missing" >&2; exit 2; }
test ! -e "$ROOT/summary.json" || {
  echo "Frozen generation result already exists; no overwrite" >&2; exit 3;
}
export PYTHONPATH="${QLORA_EXTRA_PYTHONPATH:+$QLORA_EXTRA_PYTHONPATH:}src${PYTHONPATH:+:$PYTHONPATH}"
# Leave disk headroom for GPU-heavy projects. No new full checkpoint files.
free_kib="$(df -Pk . | awk 'NR==2 {print $4}')"
test "$free_kib" -gt 12000000 || {
  echo "Insufficient space (<12GB free); refusing model generation" >&2; exit 9;
}
(
  flock -n 9 || { echo "CodeRL-Lab GPU workload lock is held" >&2; exit 8; }
  "$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 13500
  "$PYTHON" - <<'PY'
import torch,bitsandbytes
assert torch.cuda.is_available() and torch.cuda.device_count()==1
print("TRAIN009B original SFT GPU and bnb:",torch.__version__,bitsandbytes.__version__,flush=True)
PY
  "$PYTHON" -m coderl_lab.analysis.train009b_termination_audit \
    --root "$ROOT" --data-root "$DATA" --sft-dir "$SFT"
) 9>/tmp/coderl_lab_gpu_experiment_lock
test -s "$ROOT/summary.json"
echo "TRAIN009B 16-train+16-heldout EOS-by-length audit complete"
