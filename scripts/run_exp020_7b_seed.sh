#!/usr/bin/env bash
# EXP-020: independent GRPO training seeds; preserves seed42 and existing worktrees.
set -euo pipefail
SEED="${1:?Usage: bash scripts/run_exp020_7b_seed.sh 43|44}"
case "$SEED" in 43|44) ;; *) echo "Only pre-registered seeds 43/44" >&2; exit 2;; esac
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-/home/you/projects/CodeRL-Lab/.venv-gpu/bin/python}"
TRAIN017_CONFIG="configs/train017_7b_grpo_new_algorithms.yaml"
TRAIN017_DATA="${TRAIN017_DATA:-artifacts/train017/frozen24}"
SFT_ADAPTER="${SFT_ADAPTER:-/home/you/projects/CodeRL-Lab-track-train011/artifacts/train011/qlora-ultrachat}"
OUT="artifacts/train017/grpo7b_formal_seed${SEED}"
CONFIG="artifacts/train017/exp020_seed${SEED}_config.yaml"
LOG="artifacts/train017/exp020_seed${SEED}_train.log"
test -x "$PYTHON"
test -f "$TRAIN017_CONFIG"
test -s "$SFT_ADAPTER/adapter_model.safetensors"
test -s "$TRAIN017_DATA/manifest.json"
test ! -e "$OUT" || { echo "Output exists: refusing overwrite" >&2; exit 3; }
test ! -e "$LOG" || { echo "Log exists: refusing overwrite" >&2; exit 3; }
mkdir -p artifacts/train017
"$PYTHON" - "$TRAIN017_CONFIG" "$CONFIG" "$SEED" "$OUT" <<'PY'
import sys,yaml,pathlib
src,dst,seed,out=sys.argv[1:]
c=yaml.safe_load(pathlib.Path(src).read_text())
assert c["experiment"]["id"]=="TRAIN-017"
assert c["runs"]["formal_steps"]==32
assert c["grpo"]["num_generations"]==4
c["experiment"]["seed"]=int(seed)
c["runs"]["formal_output"]=out
p=pathlib.Path(dst)
if p.exists(): raise FileExistsError(p)
p.write_text(yaml.safe_dump(c,sort_keys=False))
PY
(
 flock -n 9 || { echo "GPU lock held; aborting seed $SEED" >&2; exit 9; }
 export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=src
 "$PYTHON" -m coderl_lab.train.gpu_preflight --min-free-mib 12000
 "$PYTHON" -m coderl_lab.train.train017_7b_grpo \
  --config "$CONFIG" --sft-adapter "$SFT_ADAPTER" \
  --data-dir "$TRAIN017_DATA" --output-dir "$OUT" \
  --max-steps 32 --run-kind formal
) 9>/tmp/coderl_lab_gpu_experiment_lock > "$LOG" 2>&1
"$PYTHON" - "$OUT/summary.json" "$SEED" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
assert s["status"]=="real_GPU_GRPO_completed"
assert s["global_step"]==32
assert s["initial_SFT_parameter_elements_changed_after_grpo"]>0
print("Independent TRAIN-017 7B seed",sys.argv[2],"verified",s["saved_adapter_sha256"])
PY
