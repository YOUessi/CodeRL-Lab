#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT="${BASELINE_ROOT:-artifacts/exp006a}"
PROCESS_ROOT="${PROCESS_ROOT:-artifacts/exp004a}"

python -m coderl_lab.analysis.execution_stage_diagnostics \
  --tasks data/generated/mbpp-v1/validation_tasks.jsonl \
  --predictions "$BASELINE_ROOT/eval-validation/grpo/predictions.jsonl" \
  --details "$BASELINE_ROOT/eval-validation/grpo/evaluation/details.jsonl" \
  --output "$PROCESS_ROOT/compare/baseline_diagnostics.json"

python -m coderl_lab.analysis.execution_stage_diagnostics \
  --tasks data/generated/mbpp-v1/validation_tasks.jsonl \
  --predictions "$PROCESS_ROOT/eval-validation/process_grpo/predictions.jsonl" \
  --details "$PROCESS_ROOT/eval-validation/process_grpo/evaluation/details.jsonl" \
  --output "$PROCESS_ROOT/compare/process_diagnostics.json"

python -m coderl_lab.analysis.paired_bootstrap \
  --a "$BASELINE_ROOT/eval-validation/grpo/evaluation/summary.json" \
  --b "$PROCESS_ROOT/eval-validation/process_grpo/evaluation/summary.json" \
  --iterations 20000 \
  --seed 42 \
  --output "$PROCESS_ROOT/compare/paired_bootstrap.json"

python -m coderl_lab.analysis.process_reward_compare \
  --baseline-dynamics "$BASELINE_ROOT/grpo-qwen3-1.7b/dynamics_summary.json" \
  --process-dynamics "$PROCESS_ROOT/grpo-process-qwen3-1.7b/dynamics_summary.json" \
  --baseline-eval "$BASELINE_ROOT/eval-validation/grpo/evaluation/enhanced_summary.json" \
  --process-eval "$PROCESS_ROOT/eval-validation/process_grpo/evaluation/enhanced_summary.json" \
  --baseline-diagnostics "$PROCESS_ROOT/compare/baseline_diagnostics.json" \
  --process-diagnostics "$PROCESS_ROOT/compare/process_diagnostics.json" \
  --output "$PROCESS_ROOT/compare/summary.json"
