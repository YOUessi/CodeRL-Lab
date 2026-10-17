#!/usr/bin/env bash
# EXP-023: run external MBPP+ 96-task test AFTER the matching arm finished.
# Usage on Tang: bash scripts/run_exp023_mbpp96_arm.sh execution|full
set -euo pipefail
ARM="${1:?execution|full}"
case "$ARM" in execution|full) ;; *) echo "only execution or full needed; format adapter unchanged" >&2;exit 2;; esac
WORKSPACE="/home/you/projects/CodeRL-Lab-track-exp019b"
TRAIN="/home/you/projects/CodeRL-Lab-track-train017"
PY="/home/you/projects/CodeRL-Lab/.venv-gpu/bin/python"
"$PY" "$TRAIN/scripts/exp023_prepare_mbpp96_eval.py" --workspace "$WORKSPACE" --train-root "$TRAIN" --arm "$ARM"
cd "$WORKSPACE"
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
SOURCE="artifacts/exp021/frozen96"
OUT="artifacts/exp023_eval/${ARM}_mbpp96"
mkdir -p artifacts/exp023_eval
test -s "$SOURCE/manifest.json" || exit 4
if test ! -s "$OUT/generation_summary.json";then
 (
  flock -n 9 || { echo "GPU busy; fail closed" >&2;exit 9; }
  "$PY" -m coderl_lab.train.gpu_preflight --min-free-mib 12000
  "$PY" -m "coderl_lab.analysis.exp023_${ARM}_generate" \
    --public "$SOURCE/public96_prompts.jsonl" \
    --manifest "$SOURCE/manifest.json" \
    --sft "/home/you/projects/CodeRL-Lab-track-train011/artifacts/train011/qlora-ultrachat" \
    --grpo "$TRAIN/artifacts/exp023/grpo7b_${ARM}_seed42" \
    --out "$OUT"
 ) 9>/tmp/coderl_lab_gpu_experiment_lock > "artifacts/exp023_eval/${ARM}_generate.log" 2>&1
fi
test -s "$OUT/generation_summary.json" || { echo "generation not frozen" >&2; exit 10; }
if test ! -s "$OUT/score_summary.json";then
 "$PY" -m "coderl_lab.analysis.exp023_${ARM}_docker_score" \
    --public "$SOURCE/public96_prompts.jsonl" \
    --private "$SOURCE/private96_tests.jsonl" \
    --manifest "$SOURCE/manifest.json" \
    --generation-dir "$OUT" --workers 2 > "artifacts/exp023_eval/${ARM}_score.log" 2>&1
fi
test -s "$OUT/score_summary.json"
echo "EXP023_MBPP96_${ARM}_COMPLETE"
