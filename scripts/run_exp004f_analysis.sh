#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-artifacts/exp004f}"
STOCHASTIC_SUMMARY="${STOCHASTIC_SUMMARY:-artifacts/exp004e/analysis/summary.json}"

python -m coderl_lab.analysis.sampling_amplification_summary   --greedy-root "$ROOT/greedy"   --logit-root "$ROOT/logits"   --stochastic-summary "$STOCHASTIC_SUMMARY"   --output "$ROOT/summary.json"
