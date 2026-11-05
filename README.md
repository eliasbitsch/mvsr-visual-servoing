# MVSR Lab – Visual Servoing with Foundation Models

ABB GoFa (CRB 15000-5) with Robotiq 2F-85 and an eye-in-hand Intel RealSense D435 in Gazebo
Harmonic. A foundation model finds the target from a text prompt, a tracker follows it on every
frame, and an IBVS controller drives the camera into the reference pose.
Team plan and FM assignment: [docs/PLAN.md](docs/PLAN.md).

Stack: Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic, OpenCV 4.12 (Python), ros2_control,
MoveIt 2, pinocchio, PyTorch (CUDA 12.4), Hugging Face transformers, Ultralytics.

## 1. Prerequisites

**Windows 11:** NVIDIA driver on Windows, WSL2 with Ubuntu 24.04, Docker Engine installed
*inside* that Ubuntu (not Docker Desktop). GPU rendering goes through Mesa's d3d12 driver
(`/dev/dxg`), CUDA through the driver libraries in `/usr/lib/wsl/lib`.
**Native Ubuntu 24.04:** Docker Engine, NVIDIA driver and nvidia-container-toolkit.

## 2. Setup

```bash
git clone <repo> mvsr-vs && cd mvsr-vs        # inside WSL: e.g. /mnt/c/git/FH/MVSRL
cp .env.example .env                           # pick the COMPOSE_FILE line for your machine
docker compose build                           # ~20 min, ~15 GB (IMAGE_TARGET=sim: ~9 GB)
docker compose up -d                           # first start builds the ROS workspace
docker compose exec dev bash
```

Native Ubuntu only: run `xhost +local:docker` once per login so windows can open.

## 3. Run

Inside the container (`docker compose exec dev bash`):

```bash
# Simulation only: Gazebo, controllers, camera topics
ros2 launch vs_bringup sim.launch.py

# Full pipeline
ros2 launch vs_bringup servo.launch.py detector:=grounding_dino prompt:="red box"

# In a second shell: watch the detections, then start servoing
ros2 run rqt_image_view rqt_image_view /vs/debug_image
ros2 service call /ibvs_node/start std_srvs/srv/Trigger
ros2 service call /ibvs_node/stop std_srvs/srv/Trigger
```

Launch arguments of `servo.launch.py`:

| Argument | Default | Values |
|---|---|---|
| `detector` | `grounding_dino` | `grounding_dino`, `owlv2`, `yolo_world`, `ollama_vlm` |
| `prompt` | `red box` | any text, objects in the world: red box, blue cylinder, yellow mug |
| `tracker` | `csrt` | `csrt`, `kcf`, `none` |
| `refiner` | `none` | `none`, `sam2` |
| `detector_period` | `1.0` | seconds between foundation model calls |
| `gui` | `true` | `false` runs Gazebo headless |
| `auto_start` | `false` | `true` starts servoing once the target is seen |

Setting the reference pose from the current view (move the arm there first):
`ros2 service call /ibvs_node/capture_reference std_srvs/srv/Trigger`.

Model weights are downloaded on first use into the `models` Docker volume.

### Vision-language models (Qwen3-VL, Gemma 4) via Ollama

```bash
docker compose --profile vlm up -d      # Ollama on port 11435, pulls the models once (~13 GB)
docker compose exec dev bash
ros2 launch vs_bringup servo.launch.py detector:=ollama_vlm prompt:="red box" detector_period:=2.0
```

The model is chosen with `VS_OLLAMA_MODEL` (default `qwen3-vl:8b-instruct`, also `gemma4:e4b`).
An Ollama that already runs on the host can be used instead by setting
`VS_OLLAMA_URL=http://localhost:11434`; do not run both, each loads its own copy into VRAM.
A VLM call takes about 1 s (10–20 s for the first one, which loads the model), the tracker
carries the control loop in between.

## 4. Structure

```
docker/                Dockerfile (stages sim, fm), entrypoint
compose*.yaml          base service + per-machine overrides (WSL, Linux GPU, Linux GUI)
docs/PLAN.md           team plan, FM assignment, milestones
ros2_ws/src/
  gofa_description/    GoFa URDF macro + meshes
  vs_bringup/          cell xacro, Gazebo world, controllers, launch files, parameters
  vs_interfaces/       ObjectFeatures.msg
  vs_perception/       detectors (FM plug-ins), trackers, mask refiners, perception node
  vs_control/          IBVS law, hand-eye transform, differential kinematics, safety
```

## 5. Topics

| Topic | Type | Note |
|---|---|---|
| `/camera/camera/color/image_raw` | `sensor_msgs/Image` | 640x480 @ 30 Hz |
| `/camera/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | 32FC1, metres |
| `/vs/features` | `vs_interfaces/ObjectFeatures` | centroid, area, angle, box |
| `/vs/debug_image` | `sensor_msgs/Image` | detections drawn |
| `/ibvs_node/error`, `/ibvs_node/camera_twist`, `/ibvs_node/converged` | | for the evaluation |
| `/arm_velocity_controller/commands` | `std_msgs/Float64MultiArray` | joint velocities |
| `/gt/<object>/pose` | `geometry_msgs/PoseStamped` | ground truth, **evaluation only** |

## 6. Sources and licences

- GoFa meshes and kinematics: ROS-Industrial `abb_crb15000_support` (Apache 2.0); inertials added by us.
- Robotiq 2F-85: PickNik `ros2_robotiq_gripper` (BSD-3), RealSense D435: Intel `realsense-ros` (Apache 2.0).
- Grounding DINO (IDEA-Research, Apache 2.0), OWLv2 (Google, Apache 2.0) via Hugging Face transformers.
- YOLO-World and SAM 2 wrappers: Ultralytics (AGPL-3.0); SAM 2 weights by Meta (Apache 2.0).
- Qwen3-VL (Alibaba, Apache 2.0) and Gemma 4 (Google, Apache 2.0) served by Ollama (MIT).
- IBVS features: O. Tahri, F. Chaumette, "Point-based and region-based image moments for visual
  servoing of planar objects", IEEE T-RO 2005.
