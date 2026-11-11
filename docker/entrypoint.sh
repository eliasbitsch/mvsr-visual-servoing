#!/bin/bash
set -e
source /opt/ros/jazzy/setup.bash
source /ws/install/setup.bash
if [ -f /ws/student/install/setup.bash ]; then
  source /ws/student/install/setup.bash
fi
exec "$@"
