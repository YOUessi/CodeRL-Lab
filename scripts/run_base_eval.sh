#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B-Base}"
TASKS="${TASKS:-benchmarks/sample_tasks.jsonl}"
PREDICTIONS="${PREDICTIONS:-artifacts/exp001/predictions.jsonl}"
GENERATION_META="${GENERATION_META:-artifacts/exp001/generation_meta.json}"
OUTPUT="${OUTPUT:-results/exp001-base}"
DOCKER_IMAGE="${DOCKER_IMAGE:-python:3.11-slim}"

if ! docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
  echo "Preparing Docker execution image: $DOCKER_IMAGE"
  docker pull "$DOCKER_IMAGE"
fi

python -m coderl_lab.generation \
  --tasks "$TASKS" \
  --output "$PREDICTIONS" \
  --metadata-output "$GENERATION_META" \
  --model "$MODEL" \
  --num-samples 16 \
  --max-new-tokens 512 \
  --temperature 0.8 \
  --top-p 0.95 \
  --seed 42

python -m coderl_lab.evaluation \
  --tasks "$TASKS" \
  --predictions "$PREDICTIONS" \
  --output "$OUTPUT" \
  --executor docker \
  --timeout 3 \
  --memory 512m \
  --k 1 4 8 16
