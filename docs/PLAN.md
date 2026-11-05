# MVSR Lab – Visual Servoing with Foundation Models

Team: Elias, Philip, Jiayi · Variant: **FM** · Spec: `moodle/4_Projektangaben/1_Angabe Visual Servoing.pdf`

## Goal in one sentence

An eye-in-hand RGB-D camera on a robot arm finds an object described in natural language
("the red box") with a foundation model, and an image-based visual servoing (IBVS) loop drives
the camera into a desired pose above that object, using only image information for control.

## Open points to clarify with the lecturer (first online session)

1. **Robot:** the spec says a Universal Robots arm in a simulation provided via Moodle. We build
   on an **ABB GoFa (CRB 15000-5)** with a **Robotiq 2F-85** and an **Intel RealSense D435**.
   This is a deviation from the spec and needs explicit approval. The stack is robot-agnostic
   (only `vs_bringup` and `gofa_description` know the robot), so switching to the UR model from
   Moodle is a config change, not a rewrite.
2. **Group of three:** only one such group is allowed per course, confirm it is ours.
3. Which start configurations / target objects the lecturer wants to see in the evaluation.

## Architecture

```
            Gazebo Harmonic (vs_bringup)                          ROS 2 Jazzy
 ┌──────────────────────────────────────────────┐
 │ GoFa + Robotiq 2F-85 + RealSense D435 (sim)  │
 │ gz_ros2_control: joint_state_broadcaster,    │
 │   arm_velocity_controller, arm_trajectory_   │
 │   controller, gripper_controller             │
 └──────┬──────────────────────────▲────────────┘
        │ /camera/camera/color/*    │ /arm_velocity_controller/commands
        ▼                           │
 ┌──────────────────────┐   ┌───────┴────────────────────────────┐
 │ vs_perception        │   │ vs_control (ibvs_node)             │
 │  Detector (FM, slow, │──►│  features → IBVS law → v_cam       │
 │   own thread)        │   │  hand-eye adjoint → v_ee           │
 │  Tracker (fast,      │   │  pinocchio Jacobian, DLS → q̇      │
 │   every frame)       │   │  velocity + joint-limit clamping   │
 │  MaskRefiner (opt.)  │   │  safe stop on lost/stale target    │
 └──────────────────────┘   └────────────────────────────────────┘
        │ /vs/features (vs_interfaces/ObjectFeatures)
        ▼
 vs_evaluation (to do): runs N start configs, uses /gt/* only for metrics
```

ROS packages in `ros2_ws/src`:

| Package | Content | Owner |
|---|---|---|
| `gofa_description` | GoFa URDF macro with inertials, meshes from ROS-Industrial (Apache 2.0) | Elias |
| `vs_bringup` | Robot cell xacro, Gazebo world, controllers, launch files | Elias |
| `vs_interfaces` | `ObjectFeatures.msg` | – |
| `vs_perception` | FM plug-in framework, trackers, feature extraction, perception node | Elias (core), everyone (adapters) |
| `vs_control` | IBVS node, hand-eye transform, differential kinematics, safety | Philip |
| `vs_evaluation` | Evaluation runner, metrics, plots | Jiayi |

## Foundation model assignment

Every FM is one file in `vs_perception/vs_perception/detectors/`. All three of us implement the
same `Detector` interface, so the node, the tracker and the controller never change.

| Person | GPU | Foundation models | Why this split |
|---|---|---|---|
| **Elias** | RTX 3080 Ti Laptop, 16 GB | **Grounding DINO** (main detector) + **SAM 2** (mask refiner, later video tracking) | Heaviest models, needs the VRAM; also runs the live demo |
| **Philip** | RTX 3050, ~4 GB | **YOLO-World** (+ YOLOE if time) | Real-time open-vocabulary detector that fits 4 GB; baseline for "fast but less precise" |
| **Jiayi** | to check | **OWLv2** + **Florence-2** | Two different families (CLIP-style vs. generative VLM grounding), good material for the comparison chapter |

VLMs as detector (`detector:=vlm`, served by the compose service `ollama`) show the
latency extreme: the tracker has to carry the loop. First test on a synthetic image (3080 Ti,
after warm-up): Qwen3-VL-8B 0.8 s, Gemma 4 E4B 0.6 s, Qwen3.5-9B 1.1 s per call, boxes within a
few pixels; the first call loads the model (11–17 s). Candidates that fit 16 GB VRAM:

| VLM | Size | Note |
|---|---|---|
| Gemma 4 E4B | ~8B total, 4.5B effective | returns boxes as JSON (`box_2d`) without special prompting, Apache 2.0 |
| Qwen3-VL-8B (or 4B) | 8B / 4B dense | boxes in 0–1000 normalised coordinates |
| Qwen3.8-27B | 27B dense, vision | 4-bit only, ~15–16 GB, so it cannot share the GPU with the rest of the pipeline |

| GLM-4.6V-Flash | 9B, open weights (GGUF) | boxes as `<\|begin_of_box\|>[x1,y1,x2,y2]<\|end_of_box\|>`, parser done; still to test whether Ollama runs its vision part |

Not local: Gemma 4 26B-A4B (~18 GB at 4-bit), Qwen3.8-2.4T, GLM-4.5V (106B MoE);
GLM-5V-Turbo has grounding but its weight status is unclear.
Ollama keeps one model in VRAM at a time (`OLLAMA_MAX_LOADED_MODELS=1`), so compare VLMs one
after the other; Qwen3-VL-8B needs 5.4 GiB VRAM.

### Common comparison protocol (required by the spec)

So that the numbers are comparable, every FM owner reports on the **same recorded image set**:

1. Record ~200 frames from the simulation with several objects, distances and lighting
   (`ros2 bag record`), ground-truth boxes come from projecting the `/gt/*` object poses.
2. Prompts per object, at least: plain name ("box"), name + colour ("red box"),
   descriptive ("a small red cube on the floor"), wrong/ambiguous prompt (negative test).
3. Report per model × prompt: detection rate, mean IoU to ground truth, false positives,
   latency mean / p95 on the 3080 Ti, VRAM.
4. Closed-loop: servoing success rate and convergence time with detector periods
   0.2 s / 1 s / 5 s, with and without tracker.

## Work split beyond the FMs

- **Elias:** Docker image, simulation, perception node (threading, tracker re-init with frame
  buffer), integration, demo video.
- **Philip:** IBVS law and interaction matrix, hand-eye adjoint, Jacobian / damped least
  squares, velocity and joint limits, safe stop, gain study (λ) for the report.
- **Jiayi:** evaluation runner (start configs, automatic runs, success criterion), metrics and
  plots required in spec section 2.5 (error over time, velocities, camera trajectory, feature
  trajectory in the image), robustness tests (occlusion, lighting, distance).
- **Everyone:** README, slides, plagiarism-safe source list (every borrowed model, mesh and
  snippet cited).

Safety rules are enforced in `vs_control` (self-collision, workspace, floor clearance, limits,
safe stop), see [ROBOT.md](ROBOT.md). Philip owns them together with the controller.

### Optional, only once servoing and evaluation are done

An LLM agent on top ("pick the red box and put it next to the mug"): a language model plans
with a small set of skills (`servo_to(prompt)`, `open_gripper`, `close_gripper`,
`move_to_named_pose`), each skill calls the existing nodes, and the safety layer in
`vs_control` still has the last word. Beyond the assignment; worth it only as an extra in the
demo.

## Milestones

| When | Done means |
|---|---|
| Online session 1 | Container builds on all three laptops, sim runs, camera image visible, robot moves with a trajectory command. Each person has their FM detecting objects on saved sim images. Lecturer approval for GoFa. |
| Online session 2 | Closed loop works with Grounding DINO + tracker for x/y/z/yaw from one start pose. Other FMs plugged into the node. Evaluation runner can reset the robot and run one trial automatically. |
| Online session 3 | All start configurations evaluated, FM comparison table, latency study, failure cases collected. Slides draft. |
| Day before presentation | Zip submitted (code, README, slides, demo video), clean install tested on a fresh machine from the README alone. |

## How to add a foundation model

1. Create `vs_perception/vs_perception/detectors/my_model.py`, subclass `Detector`, decorate with
   `@register_detector("my_model")`, implement `load()` and `detect(image_rgb, prompt)`.
2. Import it in `detectors/__init__.py`.
3. Run `ros2 launch vs_bringup servo.launch.py detector:=my_model prompt:="red box"`.

Details on the interface: `ros2_ws/src/vs_perception/README.md`.
