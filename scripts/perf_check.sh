#!/bin/bash
# Snapshot of the running stack, for bringing components up one at a time:
# CPU/RAM per container, GPU, the simulation's real-time factor, camera and feature rates.
#   scripts/perf_check.sh [seconds]      (default 8 s of topic sampling)
secs=${1:-8}

echo "== containers"
docker stats --no-stream --format '{{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}' | grep -E 'vs-|mvsrl' | column -t

echo "== GPU"
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,clocks.sm,power.draw,clocks_throttle_reasons.active \
  --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

if docker ps --format '{{.Names}}' | grep -q '^vs-lab$'; then
  echo "== simulation"
  docker exec vs-lab bash -c "source /opt/ros/jazzy/setup.bash; timeout 6 gz topic -e -t /stats -n 1 \
    | grep -m1 real_time_factor" || echo "no /stats: simulation not running"
  probe=vs-lab
  docker ps --format '{{.Names}}' | grep -q '^vs-fm$' && probe=vs-fm
  echo "== topics (${secs} s, from $probe)"
  docker exec -i $probe bash -c "source /opt/ros/jazzy/setup.bash; \
    [ -f /ws/student/install/setup.bash ] && source /ws/student/install/setup.bash; python3 - $secs" <<'EOF'
import sys, time, importlib, rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo
rclpy.init()
node = rclpy.create_node('perf_probe')
stamps = {'camera': [], 'features': []}
node.create_subscription(CameraInfo, '/camera/camera_info',
                         lambda m: stamps['camera'].append(time.monotonic()), qos_profile_sensor_data)
try:
    ObjectFeatures = importlib.import_module('vs_interfaces.msg').ObjectFeatures
    node.create_subscription(ObjectFeatures, '/vs/features',
                             lambda m: stamps['features'].append(time.monotonic()), 50)
except ImportError:
    stamps.pop('features')
end = time.monotonic() + float(sys.argv[1])
while time.monotonic() < end:
    rclpy.spin_once(node, timeout_sec=0.05)
for name, ts in stamps.items():
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    print(f'{name:9s} {len(ts) / float(sys.argv[1]):5.1f} Hz   max gap '
          + (f'{max(gaps):.2f} s' if gaps else '-'))
EOF
fi
