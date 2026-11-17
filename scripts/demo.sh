#!/bin/bash
# Lab demo: simulation (headless), perception + IBVS, Lichtblick and the marimo app
# notebooks/demo.py. Run from WSL or Linux:
#   scripts/demo.sh [detector] [tracker]      # default grounding_dino sam2
#   scripts/demo.sh stop
set -e
cd "$(dirname "$0")/.."
export COMPOSE_FILE=${COMPOSE_FILE:-docker-compose.yml:docker-compose.fm.yml:docker-compose.wsl.yml}

if [ "${1:-}" = stop ]; then
  docker stop -t 15 vs-marimo vs-fm vs-lab >/dev/null 2>&1 || true
  docker compose stop foxglove lichtblick
  exit 0
fi
detector=${1:-grounding_dino}
tracker=${2:-sam2}
# 10 Hz instead of the evaluation's 15: on a laptop GPU, rendering under WSL (d3d12) plus SAM 2
# on every frame ran into thermal throttling and features arrived with gaps
camera_rate=${CAMERA_RATE:-10}

docker rm -f vs-marimo vs-fm vs-lab >/dev/null 2>&1 || true
echo "starting the simulation ..."
docker compose run -d --name vs-lab sim bash -lc \
  "ros2 launch vs_sim vs_sim.launch.py target:=red_can gazebo_gui:=false image_view:=false camera_rate:=$camera_rate > /tmp/launch.log 2>&1" >/dev/null
until docker exec vs-lab grep -q 'Configured and activated joint_state_broadcaster' /tmp/launch.log 2>/dev/null; do sleep 2; done
# Demo only, at runtime (the world file stays as given): 2 ms physics steps bring the real-time
# factor from ~0.5 to 1.0 on a laptop, and the sun's shadow maps are not needed for the task
docker exec -i vs-lab bash <<'EOF'
source /opt/ros/jazzy/setup.bash
svc() { gz service -s /world/vs_world/$1 --reqtype $2 --reptype gz.msgs.Boolean --timeout 5000 --req "$3" >/dev/null; }
svc set_physics gz.msgs.Physics 'max_step_size: 0.002, real_time_factor: 1.0'
svc light_config gz.msgs.Light 'name: "sun", type: DIRECTIONAL, pose: {position: {z: 10}}, diffuse: {r: 0.85, g: 0.85, b: 0.85, a: 1}, specular: {r: 0.2, g: 0.2, b: 0.2, a: 1}, direction: {x: -0.3, y: 0.2, z: -0.9}, cast_shadows: false, intensity: 1.0'
EOF

echo "starting perception ($detector + $tracker) and the controller ..."
# VLM detectors use the host's Ollama (Vulkan), see README
docker compose run -d --name vs-fm -e VS_VLM_URL=http://localhost:11434/v1 fm bash -lc \
  "cd /ws/student && colcon build --symlink-install && source install/setup.bash && exec ros2 launch vs_bringup servo.launch.py detector:=$detector tracker:=$tracker detect_on_loss:=true" >/dev/null
until docker logs vs-fm 2>&1 | grep -q -E 'models ready|process has died'; do sleep 3; done
docker logs vs-fm 2>&1 | grep -q 'models ready' || { docker logs --tail 30 vs-fm; exit 1; }

docker compose up -d foxglove lichtblick
# Only on localhost, so no token
# Display access (as the sim service) for the optional Gazebo GUI window started from the app
docker compose run -d --name vs-marimo --no-deps -v "$PWD/notebooks:/ws/notebooks" \
  -v /tmp/.X11-unix:/tmp/.X11-unix -v /mnt/wslg:/mnt/wslg -e DISPLAY="${DISPLAY:-:0}" \
  -e WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}" -e XDG_RUNTIME_DIR=/tmp/runtime-root \
  -e GALLIUM_DRIVER=d3d12 -e MESA_D3D12_DEFAULT_ADAPTER_NAME="${VS_GPU:-NVIDIA}" \
  -e QT_QPA_PLATFORM=xcb -e QT_X11_NO_MITSHM=1 fm bash -lc \
  'mkdir -p ~/.config/marimo && printf "[runtime]\nauto_instantiate = true\n" > ~/.config/marimo/marimo.toml
   source /ws/student/install/setup.bash && cd /ws/notebooks && exec $VS_PYTHON -m marimo edit demo.py --host 127.0.0.1 --port 2718 --headless --no-token --skip-update-check' >/dev/null
until curl -s -o /dev/null localhost:2718; do sleep 1; done

echo
echo "marimo:      http://localhost:2718"
echo "Lichtblick:  http://localhost:8081/?ds=foxglove-websocket&ds.url=ws://localhost:8766"
echo "stop with:   scripts/demo.sh stop"
