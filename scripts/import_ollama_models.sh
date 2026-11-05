#!/bin/bash
# Copy an existing Ollama model folder (e.g. from a Windows install) into the compose volume.
# Loading from the WSL disk is much faster than through /mnt/c (measured ~70 s vs. a few s).
#   scripts/import_ollama_models.sh /mnt/c/Users/<you>/models/ollama
set -euo pipefail
src=${1:?usage: $0 <ollama model folder with blobs/ and manifests/>}
[ -d "$src/blobs" ] && [ -d "$src/manifests" ] || { echo "$src has no blobs/ and manifests/"; exit 1; }
cd "$(dirname "$0")/.."
volume="$(docker compose --profile vlm config --format json | python3 -c 'import sys,json; print(json.load(sys.stdin)["volumes"]["ollama"]["name"])')"
docker volume create "$volume" >/dev/null
docker run --rm -v "$src":/src:ro -v "$volume":/dst alpine sh -c 'cp -a /src/. /dst/ && du -sh /dst'
echo "imported into volume $volume"
