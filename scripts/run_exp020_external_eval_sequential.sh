#!/usr/bin/env bash
# EXP-020 2026-10-15: score seed43 only after full SHA-frozen generation,
# then generate and score seed44 under the same frozen MBPP+ 24-task cohort.
set -euo pipefail
cd /home/you/projects/CodeRL-Lab-track-exp019b
PYTHON="/home/you/projects/CodeRL-Lab/.venv-gpu/bin/python"
SFT="/home/you/projects/CodeRL-Lab-track-train011/artifacts/train011/qlora-ultrachat"
FROZEN="artifacts/exp019b/frozen24"
ROOT="artifacts/exp020"
G43="$ROOT/mbppplus_seed43_full24"
G44="$ROOT/mbppplus_seed44_full24"
mkdir -p "$ROOT"
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

# The seed43 generation was started separately. Never score if it failed.
while [ ! -s "$G43/generation_summary.json" ]; do
  if ! kill -0 335831 2>/dev/null; then
    echo "Seed43 generator ended without frozen summary: abort" >&2
    exit 10
  fi
  sleep 15
done
test ! -e "$G43/score_summary.json"
"$PYTHON" -m coderl_lab.analysis.exp020_mbppplus_seed43_docker_score \
 --public "$FROZEN/public24_prompts.jsonl" \
 --private "$FROZEN/private24_tests.jsonl" \
 --manifest "$FROZEN/manifest.json" \
 --generation-dir "$G43" --workers 2 > "$ROOT/seed43_score.log" 2>&1
test -s "$G43/score_summary.json"
echo SEED43_SCORED

test ! -e "$G44/identity.json"
(
 flock -n 9 || { echo "GPU busy: refusing concurrent evaluation" >&2; exit 9; }
 "$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 12000
 "$PYTHON" -m coderl_lab.analysis.exp020_mbppplus_seed44_generate \
  --public "$FROZEN/public24_prompts.jsonl" \
  --manifest "$FROZEN/manifest.json" \
  --sft "$SFT" \
  --grpo "/home/you/projects/CodeRL-Lab-track-train017/artifacts/train017/grpo7b_formal_seed44" \
  --out "$G44"
) 9>/tmp/coderl_lab_gpu_experiment_lock > "$ROOT/seed44_generate.log" 2>&1
test -s "$G44/generation_summary.json"
echo SEED44_GENERATION_FROZEN

"$PYTHON" -m coderl_lab.analysis.exp020_mbppplus_seed44_docker_score \
 --public "$FROZEN/public24_prompts.jsonl" \
 --private "$FROZEN/private24_tests.jsonl" \
 --manifest "$FROZEN/manifest.json" \
 --generation-dir "$G44" --workers 2 > "$ROOT/seed44_score.log" 2>&1
test -s "$G44/score_summary.json"
echo BOTH_SEEDS_MBPPPLUS_SCORED
