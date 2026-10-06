#!/usr/bin/env bash
set -euo pipefail

FULL_TRAIN="${FULL_TRAIN:-artifacts/exp003/grpo-qwen3-0.6b/run_summary.json}"
FULL_DYN="${FULL_DYN:-artifacts/exp003/grpo-qwen3-0.6b/dynamics_summary.json}"
FULL_EVAL="${FULL_EVAL:-artifacts/exp003/eval-validation/grpo/evaluation/enhanced_summary.json}"

RANDOM_TRAIN="${RANDOM_TRAIN:-artifacts/exp005/grpo-random-subset-qwen3-0.6b/run_summary.json}"
RANDOM_DYN="${RANDOM_DYN:-artifacts/exp005/grpo-random-subset-qwen3-0.6b/dynamics_summary.json}"
RANDOM_EVAL="${RANDOM_EVAL:-artifacts/exp005/eval-validation/random_control/evaluation/enhanced_summary.json}"

BOUNDARY_TRAIN="${BOUNDARY_TRAIN:-artifacts/exp005/grpo-boundary-qwen3-0.6b/run_summary.json}"
BOUNDARY_DYN="${BOUNDARY_DYN:-artifacts/exp005/grpo-boundary-qwen3-0.6b/dynamics_summary.json}"
BOUNDARY_EVAL="${BOUNDARY_EVAL:-artifacts/exp005/eval-validation/boundary/evaluation/enhanced_summary.json}"

OUTPUT="${OUTPUT:-artifacts/exp005/matched_budget_comparison.json}"

python -m coderl_lab.analysis.exp005_compare \
  --full-training "$FULL_TRAIN" \
  --full-dynamics "$FULL_DYN" \
  --full-evaluation "$FULL_EVAL" \
  --random-training "$RANDOM_TRAIN" \
  --random-dynamics "$RANDOM_DYN" \
  --random-evaluation "$RANDOM_EVAL" \
  --boundary-training "$BOUNDARY_TRAIN" \
  --boundary-dynamics "$BOUNDARY_DYN" \
  --boundary-evaluation "$BOUNDARY_EVAL" \
  --rl-completions 1496 \
  --screening-completions 1496 \
  --output "$OUTPUT"
