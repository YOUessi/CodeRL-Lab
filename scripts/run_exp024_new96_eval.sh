#!/usr/bin/env bash
# EXP-024 Tang-only sequential new96 evaluation; never open private tests until SHA-frozen generation.
set -euo pipefail
TRAIN=/home/you/projects/CodeRL-Lab-track-train017
WORK=/home/you/projects/CodeRL-Lab-track-exp019b
PY=/home/you/projects/CodeRL-Lab/.venv-gpu/bin/python
export PYTHONPATH="$WORK/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
FROZEN="$WORK/artifacts/exp024/frozen_new96"
test -s "$FROZEN/manifest.json"
for seed in 43 44;do
 CHECKPOINT="$TRAIN/artifacts/exp024/strict_seed${seed}"
 # Wait for completion, do not reinterpret a partial adapter as a completed model.
 while ! test -s "$CHECKPOINT/summary.json";do
   if ! pgrep -f "[p]ython -m coderl_lab.train.exp024_7b_grpo" >/dev/null;then
     echo "Training ended without finalized checkpoint seed $seed" >&2;exit 20
   fi
   sleep 20
 done
 "$PY" "$TRAIN/scripts/exp024_prepare_new96_eval.py" --workspace "$WORK" --train-root "$TRAIN" --seed "$seed"
 cd "$WORK"
 OUT="artifacts/exp024/eval_seed${seed}_new96"
 mkdir -p artifacts/exp024
 if ! test -s "$OUT/generation_summary.json";then
  (
   flock -x 9  # wait for the 7B training GPU lock to become available
   "$PY" -m coderl_lab.train.gpu_preflight --min-free-mib 12000
   "$PY" -m "coderl_lab.analysis.exp024_seed${seed}_generate" \
    --public "artifacts/exp024/frozen_new96/public96_prompts.jsonl" \
    --manifest "artifacts/exp024/frozen_new96/manifest.json" \
    --sft "/home/you/projects/CodeRL-Lab-track-train011/artifacts/train011/qlora-ultrachat" \
    --grpo "$CHECKPOINT" --out "$OUT"
  ) 9>/tmp/coderl_lab_gpu_experiment_lock > "artifacts/exp024/eval_seed${seed}_generate.log" 2>&1
 fi
 test -s "$OUT/generation_summary.json"
 if ! test -s "$OUT/score_summary.json";then
  "$PY" -m "coderl_lab.analysis.exp024_seed${seed}_docker_score" \
   --public "artifacts/exp024/frozen_new96/public96_prompts.jsonl" \
   --private "artifacts/exp024/frozen_new96/private96_tests.jsonl" \
   --manifest "artifacts/exp024/frozen_new96/manifest.json" \
   --generation-dir "$OUT" --workers 2 > "artifacts/exp024/eval_seed${seed}_score.log" 2>&1
 fi
 test -s "$OUT/score_summary.json"
 echo "EXP024_SEED_${seed}_NEW96_SCORED"
done
