#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-artifacts/exp004l/data}"

python -m coderl_lab.datasets.livecodebench   --output-dir "$OUTPUT_DIR"

echo "Prepared LiveCodeBench v6 public-only view:"
cat "$OUTPUT_DIR/manifest.json"
