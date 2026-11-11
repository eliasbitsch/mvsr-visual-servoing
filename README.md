# MVSR Lab – Visual Servoing with Foundation Models

A UR5e with a wrist camera in the lab simulation (Gazebo Harmonic, ROS 2 Jazzy). A foundation
model finds the target from a text prompt ("red can"), a tracker follows it on every frame, and an
image-based visual servoing (IBVS) controller drives the camera into the reference pose above it.
Team plan and FM assignment: [docs/PLAN.md](docs/PLAN.md). The lab simulation itself, its
targets, start configurations and ground truth: [docs/LAB_SIM.md](docs/LAB_SIM.md).

Group C: Elias Bitsch, Philip Stix, Jiayi Zhou · Track: FM

## 1. Layout

The repository is the lecturer's template, unchanged, plus our packages and Docker files:

```
Dockerfile, docker-compose.yml,      lecturer's lab simulation (image vs-sim:jazzy)
docker-compose.novnc.yml, vs_sim/,
docker/entrypoint.sh, docker/headless_test.sh, run_headless_test.sh
docker/Dockerfile.fm                 ours: PyTorch (CUDA), OpenCV 4.12, foundation models on top of vs-sim:jazzy
docker-compose.fm.yml                ours: fm / fm-nvidia (our stack), foxglove + lichtblick (web GUI), ollama (VLM server)
docker/lichtblick-layout.json        ours: default layout of the web GUI
docker-compose.wsl.yml               ours: GPU under Windows/WSL2 (Mesa d3d12, CUDA via /usr/lib/wsl)
student_ws/src/
  vs_interfaces/                     ObjectFeatures.msg
  vs_perception/                     detectors (FM plug-ins), trackers, mask refiners, perception node
  vs_control/                        IBVS law, hand-eye transform, differential kinematics, safety
  vs_bringup/                        launch files (servo, web GUI bridge) and parameters
docs/                                plan, lab simulation docs
scripts/                             helper scripts
```

## 2. Prerequisites

- **Windows 11:** NVIDIA driver on Windows, WSL2 with Ubuntu 24.04, Docker Engine installed
  *inside* that Ubuntu (not Docker Desktop). Run every command below in a WSL shell.
- **Ubuntu 24.04:** Docker Engine, NVIDIA driver and `nvidia-container-toolkit`. Run
  `xhost +local:docker` once per login so the simulation windows can open.

## 3. Setup

```bash
git clone https://github.com/eliasbitsch/mvsr-visual-servoing.git && cd mvsr-visual-servoing
cp .env.example .env            # keep the COMPOSE_FILE line for your machine
docker compose build sim        # lab simulation, ~10 min
docker compose build fm         # our stack on top of it, ~20 min, ~12 GB
```

## 4. Run

Three terminals in the repository folder. On Ubuntu with NVIDIA use `sim-nvidia` / `fm-nvidia`
instead of `sim` / `fm`.

```bash
# 1: lab simulation with Gazebo and the camera view
docker compose run --rm sim ros2 launch vs_sim vs_sim.launch.py target:=red_can camera_rate:=15

# 2: start perception + IBVS (builds student_ws on start)
docker compose run --rm fm

# 3: take the reference features at the reference pose, move to a start configuration, servo
docker compose run --rm sim ros2 run vs_sim go_to_start --target red_can --reference
docker compose run --rm fm ros2 service call /ibvs_node/capture_reference std_srvs/srv/Trigger
docker compose run --rm sim ros2 run vs_sim go_to_start --target red_can --start 1
docker compose run --rm fm ros2 service call /ibvs_node/start std_srvs/srv/Trigger
```

### Web GUI (Lichtblick)

```bash
docker compose up -d foxglove lichtblick     # bridge on ws://localhost:8766, GUI on :8081
```

Then open <http://localhost:8081/?ds=foxglove-websocket&ds.url=ws://localhost:8766>. The layout
(`docker/lichtblick-layout.json`) loads by default, in dark mode: the UR5e and the objects of the
lab world in 3D, the camera with the detection, plots of the image feature error, the ground-truth
pose error and the commanded camera velocity, a converged timeline, and buttons for *Capture
reference*, *Start servoing* and *Stop*. With the GUI the simulation can run with
`gazebo_gui:=false`. If an older layout shows up, delete the stored layout in Lichtblick (Layouts
tab) and reload. The bridge waits until the `fm` service has built `student_ws`, so start `fm`
first. Stop the GUI with `docker compose stop foxglove lichtblick`; nothing here starts on its own.

`camera_rate:=15` is enough for our perception (it handles about 12 frames per second) and halves
the rendering load. Add `gazebo_gui:=false` while working on the controller: the Gazebo window
alone costs about 1.4 CPU cores, the camera view still opens.

Under WSL keep the terminals open: WSL shuts its VM down when no WSL session is left, and the
containers then exit with code 255.

Stopping `fm` (Ctrl+C or `docker compose stop`) also stops the arm: the IBVS node sends zero joint
velocities on shutdown, since the lab's velocity controller would keep the last command.

### Detectors, trackers and yaw

The rotation about the optical axis is servoed only with a mask that has a defined principal
axis: use `tracker:=sam2` (CSRT gives a box without angle). Round objects (can, ball) look the
same under any rotation, so their yaw stays as it started; the answer of *Capture reference* says
whether yaw is servoed. A rectangle that is 180 deg off looks identical to the reference (start
configuration S7), so that ambiguity remains.

Other prompts and detectors go after the service name, for example
`docker compose run --rm fm ros2 launch vs_bringup servo.launch.py detector:=owlv2 prompt:="blue box"`.

| Argument | Default | Values |
|---|---|---|
| `detector` | `grounding_dino` | `grounding_dino`, `owlv2`, `yolo_world`, `vlm` |
| `prompt` | `red can` | any text; FM targets in the world: red can, blue box, green cup, yellow ball |
| `tracker` | `csrt` | `csrt`, `kcf`, `sam2` (mask, needed for yaw), `none` |
| `refiner` | `none` | `none`, `sam2` |
| `detector_period` | `1.0` | seconds between foundation model calls |
| `detect_on_loss` | `false` | `true` runs the detector only while no track exists (use with `tracker:=sam2`) |
| `auto_start` | `false` | `true` starts servoing once the target is seen |

Model weights are downloaded on first use into the `models` Docker volume.

### Vision-language models (Qwen3-VL, Gemma 4) via Ollama

```bash
docker compose --profile vlm up -d ollama
docker compose exec ollama ollama pull qwen3-vl:8b-instruct
docker compose run --rm fm ros2 launch vs_bringup servo.launch.py detector:=vlm detector_period:=2.0
```

The detector speaks the OpenAI-compatible chat API, so Ollama can be swapped for vLLM, llama.cpp
server or SGLang through `VS_VLM_URL`; the model is chosen with `VS_VLM_MODEL`. Models already
downloaded elsewhere can be copied into the volume: `scripts/import_ollama_models.sh <folder>`.
A VLM call takes 1–2 s, the tracker carries the control loop in between.

## 5. Topics

| Topic | Type | Note |
|---|---|---|
| `/camera/image_raw`, `/camera/camera_info` | `sensor_msgs` | lab camera, 640x480 |
| `/vs/features` | `vs_interfaces/ObjectFeatures` | centroid, area, angle, box |
| `/vs/debug_image` | `sensor_msgs/Image` | detections drawn |
| `/ibvs_node/error`, `/ibvs_node/camera_twist`, `/ibvs_node/converged` | | for the evaluation |
| `/forward_velocity_controller/commands` | `std_msgs/Float64MultiArray` | joint velocities (control output) |
| `/ground_truth/*` | | lab ground truth, **evaluation only**, never used in control |

## 6. Sources and licences

- Lab simulation (`vs_sim`, Dockerfile, compose files): course material, TW-Robotics MVSR.
- Grounding DINO (IDEA-Research, Apache 2.0), OWLv2 (Google, Apache 2.0) via Hugging Face transformers.
- YOLO-World and SAM 2 wrappers: Ultralytics (AGPL-3.0); SAM 2 weights by Meta (Apache 2.0).
- Qwen3-VL (Alibaba, Apache 2.0) and Gemma 4 (Google, Apache 2.0) served by Ollama (MIT).
- IBVS features: O. Tahri, F. Chaumette, "Point-based and region-based image moments for visual
  servoing of planar objects", IEEE T-RO 2005.
