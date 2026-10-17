#!/usr/bin/env bash
# Full bounded chain, Tang only. Run from a pinned GitHub checkout.
set -euo pipefail
cd "$(dirname "$0")/.."
[ "$(hostname)" = Tang ] || { echo 'Tang required' >&2; exit 2; }
export PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
PY=/home/you/projects/CodeRL-Lab/.venv-gpu/bin/python
TRAIN=/home/you/projects/CodeRL-Lab-track-train017
EVAL=/home/you/projects/CodeRL-Lab-track-exp019b
OUT="$TRAIN/artifacts/exp026"
mkdir -p "$OUT"
"$PY" -m coderl_lab.analysis.exp026_runner --train-root "$TRAIN" --eval-root "$EVAL" --out "$OUT"
for seed in 43 44; do
  "$PY" - "$TRAIN" "$seed" <<'PY'
import sys,json,yaml
from pathlib import Path
root=Path(sys.argv[1]);seed=int(sys.argv[2]);c=yaml.safe_load((root/f'configs/exp024_strict_seed{seed}.yaml').read_text())
c['runs']['formal_output']=str(root/f'artifacts/exp026/depfix_seed{seed}')
c['runs']['smoke_output']=str(root/f'artifacts/exp026/smoke_seed{seed}')
p=root/f'artifacts/exp026/seed{seed}_config.yaml'
text=yaml.safe_dump(c,sort_keys=False)
if p.exists() and p.read_text()!=text:raise ValueError('config changed')
if not p.exists():p.write_text(text)
PY
  if [ ! -s "$OUT/depfix_seed${seed}/summary.json" ]; then
    (
      flock -n 9 || { echo 'GPU busy; abort without concurrent training' >&2; exit 9; }
      "$PY" -m coderl_lab.train.gpu_preflight --min-free-mib 12000
      "$PY" -m coderl_lab.train.exp026_7b_grpo --arm strict \
        --config "$OUT/seed${seed}_config.yaml" --data-dir "$TRAIN/artifacts/train017/frozen24" \
        --sft-adapter /home/you/projects/CodeRL-Lab-track-train011/artifacts/train011/qlora-ultrachat \
        --output-dir "$OUT/depfix_seed${seed}" --max-steps 32 --run-kind formal
    ) 9>/tmp/coderl_lab_gpu_experiment_lock > "$OUT/train_seed${seed}.log" 2>&1
  fi
  echo "EXP026_TRAINING_COMPLETED_SEED_$seed"
done
"$PY" -m coderl_lab.analysis.exp026_runner --train-root "$TRAIN" --eval-root "$EVAL" --out "$OUT" --post-train
echo EXP026_CHAIN_COMPLETE
