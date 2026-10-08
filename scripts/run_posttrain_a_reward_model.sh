#!/usr/bin/env bash
# TRAIN-008A: verified frozen UltraFeedback paired reward-model training.
# MODE=smoke uses 32/16 pairs; MODE=formal uses all 2048/256.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
MODE="${MODE:-smoke}"
EXTRA_ARGS=()
case "$MODE" in
  smoke)
    OUTPUT="artifacts/posttrain-a/train008a-reward-model-smoke"
    EXTRA_ARGS+=(--smoke-train-pairs 32 --smoke-eval-pairs 16)
    ;;
  formal)
    OUTPUT="artifacts/posttrain-a/train008a-reward-model"
    ;;
  *) echo "MODE must be smoke or formal" >&2; exit 2 ;;
esac
test -x "$PYTHON" || { echo "Python environment missing" >&2; exit 2; }
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" - <<'PY'
from pathlib import Path
from coderl_lab.train.reward_model import validate_preference_snapshot
cfg, train, eval_, identity = validate_preference_snapshot(
    config_path=Path("configs/posttrain_a_reward_model_qwen3_1.7b.yaml"),
    train_path=Path("data/generated/posttrain-h4-v1/dpo_train.jsonl"),
    heldout_path=Path("data/generated/posttrain-h4-v1/dpo_validation.jsonl"),
    manifest_path=Path("data/generated/posttrain-h4-v1/dpo_manifest.json"),
)
assert len(train)==2048 and len(eval_)==256 and identity["formal"] is True
print("TRAIN-008A full pinned preference manifest, data SHA and split disjointness PASS")
PY
(
  flock -n 9 || { echo "CodeRL-Lab GPU experiment lock is held" >&2; exit 8; }
  "$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 12000
  ARGS=("${EXTRA_ARGS[@]}")
  if [ -n "${RESUME_CHECKPOINT:-}" ]; then
    ARGS+=(--resume-from-checkpoint "$RESUME_CHECKPOINT")
  fi
  "$PYTHON" -m coderl_lab.train.reward_model \
    --config configs/posttrain_a_reward_model_qwen3_1.7b.yaml \
    --output-dir "$OUTPUT" "${ARGS[@]}"
) 9>/tmp/coderl_lab_gpu_experiment_lock
