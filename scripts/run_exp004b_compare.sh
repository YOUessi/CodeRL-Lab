#!/usr/bin/env bash
set -euo pipefail

SFT_ROOT="${SFT_ROOT:-artifacts/exp006b/n16/sft}"
DPO_ROOT="${DPO_ROOT:-artifacts/exp004b/eval-validation/dpo}"
OUT="${OUT:-artifacts/exp004b/compare}"

mkdir -p "$OUT"

python -m coderl_lab.analysis.execution_stage_diagnostics \
  --tasks data/generated/mbpp-v1/validation_tasks.jsonl \
  --predictions "$SFT_ROOT/predictions.jsonl" \
  --details "$SFT_ROOT/evaluation/details.jsonl" \
  --output "$OUT/sft_diagnostics.json"

python -m coderl_lab.analysis.execution_stage_diagnostics \
  --tasks data/generated/mbpp-v1/validation_tasks.jsonl \
  --predictions "$DPO_ROOT/predictions.jsonl" \
  --details "$DPO_ROOT/evaluation/details.jsonl" \
  --output "$OUT/dpo_diagnostics.json"

python -m coderl_lab.analysis.passk_bootstrap \
  --a "$SFT_ROOT/evaluation/summary.json" \
  --b "$DPO_ROOT/evaluation/summary.json" \
  --k 1 4 8 16 \
  --iterations 20000 \
  --seed 42 \
  --output "$OUT/passk_bootstrap.json"

python -m coderl_lab.analysis.dpo_repair_compare \
  --sft-eval "$SFT_ROOT/evaluation/enhanced_summary.json" \
  --dpo-eval "$DPO_ROOT/evaluation/enhanced_summary.json" \
  --sft-diagnostics "$OUT/sft_diagnostics.json" \
  --dpo-diagnostics "$OUT/dpo_diagnostics.json" \
  --passk-bootstrap "$OUT/passk_bootstrap.json" \
  --output "$OUT/summary.json"
