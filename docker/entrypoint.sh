#!/bin/bash
set -e
source /opt/ros/jazzy/setup.bash

# Build the mounted workspace on first start so `docker compose up` works without manual steps.
if [ -d "${ROS_WS}/src" ] && [ ! -f "${ROS_WS}/install/setup.bash" ]; then
  echo "[entrypoint] building workspace ..."
  (cd "${ROS_WS}" && colcon build --symlink-install)
fi
[ -f "${ROS_WS}/install/setup.bash" ] && source "${ROS_WS}/install/setup.bash"

exec "$@"
