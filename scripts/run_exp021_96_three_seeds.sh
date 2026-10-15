#!/usr/bin/env bash
# EXP-021 sequential 96-task MBPP+ multi-seed evaluation on Tang only.
set -euo pipefail
cd /home/you/projects/CodeRL-Lab-track-exp019b
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
PY=/home/you/projects/CodeRL-Lab/.venv-gpu/bin/python
P=artifacts/exp021/frozen96
SFT=/home/you/projects/CodeRL-Lab-track-train011/artifacts/train011/qlora-ultrachat
ROOT=artifacts/exp021
run_score() {
  local seed="$1" dir="$ROOT/eval_seed${seed}_96"
  test -s "$dir/generation_summary.json" || { echo "No SHA-frozen generation for $seed" >&2; return 4; }
  if test -e "$dir/score_summary.json"; then echo "Already scored $seed, preserving";return 0; fi
  "$PY" -m coderl_lab.analysis.exp021_seed${seed}_docker_score --public "$P/public96_prompts.jsonl" --private "$P/private96_tests.jsonl" --manifest "$P/manifest.json" --generation-dir "$dir" --workers 2 > "$ROOT/seed${seed}_score.log" 2>&1
  test -s "$dir/score_summary.json"
  echo "Scored seed $seed"
}
# Existing seed42 generation was started separately; do not claim success from PID.
while ! test -s "$ROOT/eval_seed42_96/generation_summary.json"; do
  if ! pgrep -f '[p]ython -m coderl_lab.analysis.exp021_seed42_generate' >/dev/null; then
    echo "seed42 ended without successful full generation; abort" >&2;exit 10
  fi
  sleep 15
done
run_score 42
for seed in 43 44;do
 dir="$ROOT/eval_seed${seed}_96"
 if ! test -s "$dir/generation_summary.json"; then
  (
   flock -n 9 || { echo "GPU locked: refusing concurrency" >&2;exit 9; }
   "$PY" -m coderl_lab.train.gpu_preflight --min-free-mib 12000
   "$PY" -m coderl_lab.analysis.exp021_seed${seed}_generate --public "$P/public96_prompts.jsonl" --manifest "$P/manifest.json" --sft "$SFT" --grpo "/home/you/projects/CodeRL-Lab-track-train017/artifacts/train017/grpo7b_formal_seed${seed}" --out "$dir"
  ) 9>/tmp/coderl_lab_gpu_experiment_lock > "$ROOT/seed${seed}_generate.log" 2>&1
 fi
 run_score "$seed"
done
echo EXP021_ALL_THREE_SEEDS_SCORED
