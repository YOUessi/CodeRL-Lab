#!/usr/bin/env bash
# TRAIN-017: frozen 24-task/two-seed/three-arm Qwen2.5-7B evaluation.
# The generate action NEVER opens a heldout answer oracle.
set -euo pipefail
cd "$(dirname "$0")/.."
PHASE="${1:?generate|score}"
ROOT="artifacts/train017"
DATA="$ROOT/frozen24"
OUT="$ROOT/fresh24_3arm_2seed_n4"
PYTHON="${PYTHON:-/home/you/projects/CodeRL-Lab/.venv-gpu/bin/python}"
SFT="${SFT:-/home/you/projects/CodeRL-Lab-track-train011/artifacts/train011/qlora-ultrachat}"
PREV="${PREV:-/home/you/projects/CodeRL-Lab-track-train016/artifacts/train016/grpo7b_formal_v1}"
NEW="$ROOT/grpo7b_formal_v1"
if [ "$PHASE" = "generate" ]; then
  (
    flock -n 9 || { echo "Tang GPU already in use" >&2;exit 9; }
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
    PYTHONPATH=src "$PYTHON" -m coderl_lab.analysis.train017_7b_generate \
      --public "$DATA/holdout_public.jsonl" \
      --manifest "$DATA/manifest.json" \
      --sft "$SFT" --previous "$PREV" --new "$NEW" \
      --new-summary "$NEW/summary.json" --output-dir "$OUT"
  ) 9>/tmp/coderl_lab_gpu_experiment_lock
elif [ "$PHASE" = "score" ]; then
  test -s "$OUT/generation_summary.json" || {
    echo "All 576 generated outputs MUST be frozen before private oracle scoring" >&2;exit 4;
  }
  test ! -e "$OUT/docker_score_summary.json" || {
    echo "Already finalized; refusing to overwrite evaluation" >&2;exit 4;
  }
  PYTHONPATH=src "$PYTHON" -m coderl_lab.analysis.train017_7b_docker_score \
      --public "$DATA/holdout_public.jsonl" \
      --oracle "$DATA/holdout_oracle.jsonl" \
      --manifest "$DATA/manifest.json" \
      --generation-dir "$OUT" --workers 4
else
  echo "Expected generate or score" >&2;exit 2
fi
