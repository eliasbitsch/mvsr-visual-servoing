#!/bin/bash
# Builds the image and runs the headless smoke test (macOS / Linux, no GUI needed).
#   ./run_headless_test.sh
# Results: data/build.log, data/test/test.log, data/test/launch.log, data/reference/*.png
# Works without bind mounts (Colima/Docker Desktop do not share external drives by default);
# results are copied out of the container with "docker cp".
set -eo pipefail
cd "$(dirname "$0")"
mkdir -p data/test
echo "Building image (first time: 10-20 min) ..."
docker build -t vs-sim:jazzy . 2>&1 | tee data/build.log
echo "Running headless test ..."
docker rm -f vs-sim-test >/dev/null 2>&1 || true
set +e
docker run --name vs-sim-test -e LIBGL_ALWAYS_SOFTWARE=1 vs-sim:jazzy bash /headless_test.sh
set -e
docker cp vs-sim-test:/root/vs_data/. data/ || true
docker rm vs-sim-test >/dev/null
echo "Done. Results in data/test/ and data/reference/"
