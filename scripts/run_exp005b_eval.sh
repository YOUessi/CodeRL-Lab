#!/usr/bin/env bash
set -euo pipefail

ARM=online_dynamic \
ADAPTER="${ADAPTER:-artifacts/exp005b/grpo-online-qwen3-0.6b}" \
ROOT="${ROOT:-artifacts/exp005b/eval-validation}" \
bash scripts/run_exp005_arm_eval.sh
