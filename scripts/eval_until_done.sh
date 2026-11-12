#!/bin/bash
# Runs the evaluation to completion. Under WSL the lab's ros_gz_bridge sometimes loses all
# Gazebo topics (/clock, camera) after a few minutes; run_eval then stops with an error, and this
# script restarts the sim container and resumes the same output folder (finished runs are kept).
#   scripts/eval_until_done.sh <label> [run_eval options]
# Needs the sim running as container vs-lab and the stack as vs-fm (see README).
set -u
label=${1:?usage: $0 <label> [run_eval options]}
shift
out=/root/vs_data/eval/$(date +%Y%m%d_%H%M%S)_$label
env='source /opt/ros/jazzy/setup.bash && source /ws/install/setup.bash && source /ws/student/install/setup.bash'

for attempt in 1 2 3 4 5 6; do
  if docker exec vs-fm bash -c "$env && ros2 run vs_evaluation run_eval --resume $out $*"; then
    echo "done after $attempt attempt(s): $out"
    exit 0
  fi
  echo "attempt $attempt stopped, restarting the sim"
  docker restart vs-lab >/dev/null
  until docker exec vs-lab grep -q 'Configured and activated joint_state_broadcaster' /tmp/launch.log 2>/dev/null; do
    sleep 2
  done
  sleep 5
done
echo "giving up after 6 attempts: $out"
exit 1
