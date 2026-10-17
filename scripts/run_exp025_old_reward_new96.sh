#!/usr/bin/env bash
set -euo pipefail
cd /home/you/projects/CodeRL-Lab-track-exp019b
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
PY=/home/you/projects/CodeRL-Lab/.venv-gpu/bin/python
DATA=artifacts/exp024/frozen_new96
for seed in 43 44; do
 OUT="artifacts/exp025/old_seed${seed}_new96"
 if ! test -s "$OUT/generation_summary.json";then
  (
    flock -x 9
    "$PY" -m coderl_lab.train.gpu_preflight --min-free-mib 12000
    "$PY" -m coderl_lab.analysis.exp025_old_seed${seed}_generate \
       --public "$DATA/public96_prompts.jsonl" --manifest "$DATA/manifest.json" \
       --sft /home/you/projects/CodeRL-Lab-track-train011/artifacts/train011/qlora-ultrachat \
       --grpo /home/you/projects/CodeRL-Lab-track-train017/artifacts/train017/grpo7b_formal_seed${seed} \
       --out "$OUT"
  ) 9>/tmp/coderl_lab_gpu_experiment_lock > "artifacts/exp025/seed${seed}_generate.log" 2>&1
 fi
 test -s "$OUT/generation_summary.json"
 if ! test -s "$OUT/score_summary.json";then
   "$PY" -m coderl_lab.analysis.exp025_old_seed${seed}_docker_score \
      --public "$DATA/public96_prompts.jsonl" --private "$DATA/private96_tests.jsonl" \
      --manifest "$DATA/manifest.json" --generation-dir "$OUT" --workers 2 \
       > "artifacts/exp025/seed${seed}_score.log" 2>&1
 fi
 test -s "$OUT/score_summary.json"
 echo "COMPLETE_SEED_${seed}"
done
