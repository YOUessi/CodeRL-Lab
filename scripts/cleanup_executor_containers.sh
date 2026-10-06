#!/usr/bin/env bash
set -euo pipefail

ids="$(docker ps -q --filter label=coderl_lab=1)"
if [ -z "$ids" ]; then
  echo "No running CodeRL-Lab executor containers."
  exit 0
fi

echo "Removing running CodeRL-Lab executor containers:"
docker ps --filter label=coderl_lab=1 --format '{{.ID}} {{.Names}} {{.Status}}'
docker rm -f $ids
