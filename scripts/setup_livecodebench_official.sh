#!/usr/bin/env bash
set -euo pipefail

PINNED_COMMIT="28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
ROOT="${ROOT:-.external/livecodebench}"
IMAGE="${IMAGE:-coderl-lab-lcb-eval}"

mkdir -p "$(dirname "$ROOT")"

if [ ! -d "$ROOT/.git" ]; then
  git clone --filter=blob:none --no-checkout     https://github.com/LiveCodeBench/LiveCodeBench.git     "$ROOT"
fi

git -C "$ROOT" fetch --depth 1 origin "$PINNED_COMMIT"
git -C "$ROOT" checkout --detach "$PINNED_COMMIT"

ACTUAL="$(git -C "$ROOT" rev-parse HEAD)"
if [ "$ACTUAL" != "$PINNED_COMMIT" ]; then
  echo "LiveCodeBench pin mismatch: expected $PINNED_COMMIT got $ACTUAL" >&2
  exit 2
fi

docker build -t "$IMAGE" docker/livecodebench-eval

echo "LiveCodeBench source: $ACTUAL"
echo "Evaluator image: $IMAGE"
