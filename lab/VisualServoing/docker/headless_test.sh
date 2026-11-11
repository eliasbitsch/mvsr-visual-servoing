#!/bin/bash
# Headless smoke test of the simulation. Runs INSIDE the container.
# Results: /root/vs_data/test/ (= data/test/ on the host)
# (no "set -u": the ROS setup scripts use unset variables)
OUT=/root/vs_data/test
mkdir -p "$OUT"
source /opt/ros/jazzy/setup.bash
source /ws/install/setup.bash
exec > >(tee "$OUT/test.log") 2>&1

step() { echo; echo "================ $* ================"; }

step "unit tests"
(cd /ws/src/vs_sim && python3 -m pytest -q -p no:cacheprovider test)

step "check_start_configs"
ros2 run vs_sim check_start_configs | tail -4

step "launch (headless)"
ros2 launch vs_sim vs_sim.launch.py gazebo_gui:=false image_view:=false > "$OUT/launch.log" 2>&1 &
LPID=$!
for i in $(seq 1 120); do
  ros2 topic list 2>/dev/null | grep -q "^/camera/image_raw$" && break
  sleep 2
done
sleep 20

step "topics"
ros2 topic list

step "controllers"
ros2 control list_controllers

step "camera rate (8 s)"
timeout -s INT 8 ros2 topic hz /camera/image_raw 2>&1 | tail -4

step "camera_info"
timeout 15 ros2 topic echo --once /camera/camera_info 2>&1 | grep -E "frame_id|height|width|^k:" -A0
timeout 15 ros2 topic echo --once /camera/camera_info --field k 2>&1 | head -12

step "ground truth model names"
timeout 15 ros2 topic echo --once /ground_truth/model_poses 2>&1 | grep child_frame_id | sort | uniq | head -40

step "capture_reference --all"
timeout 600 ros2 run vs_sim capture_reference --all

for s in 1 3 6 7; do
  step "go_to_start aruco $s"
  timeout 180 ros2 run vs_sim go_to_start --target aruco --start $s
  timeout 10 ros2 topic echo --once /ground_truth/pose_error 2>&1 | head -12
done

step "go_to_start red_can 2"
timeout 180 ros2 run vs_sim go_to_start --target red_can --start 2
timeout 10 ros2 topic echo --once /ground_truth/pose_error 2>&1 | head -12

step "example_joint_velocity"
timeout 90 ros2 run vs_sim example_joint_velocity

step "done"
ls -la /root/vs_data/reference 2>/dev/null
kill -INT $LPID 2>/dev/null; sleep 5; pkill -9 -f "gz sim" 2>/dev/null; true
