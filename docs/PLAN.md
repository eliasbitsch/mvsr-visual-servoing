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

| GLM-4.6V-Flash | 9B, open weights (GGUF Q4_K_M, 8 GB) | tested in Ollama 0.40.14: boxes correct (~2 px) in 2 of 3 identical calls, the third produced only `!!!!…` until the token limit; 27–34 s per answer (reasons first, 0.9B vision encoder). Not usable in the loop for now; to retry with Q8_0 or vLLM + AWQ |

Not local: Gemma 4 26B-A4B (~18 GB at 4-bit), Qwen3.8-2.4T, GLM-4.5V (106B MoE);
GLM-5V-Turbo has grounding but its weight status is unclear.
Ollama keeps one model in VRAM at a time (`OLLAMA_MAX_LOADED_MODELS=1`), so compare VLMs one
after the other; Qwen3-VL-8B needs 5.4 GiB VRAM.

### Common comparison protocol (required by the spec)

So that the numbers are comparable, every FM owner reports on the **same recorded image set**:

1. A clutter world `vs_world_clutter.sdf` with realistic textured objects from Gazebo Fuel and
   **distractors that only the prompt can separate**:

   | Role | Fuel model |
   |---|---|
   | drill 1 | OpenRobotics "Cordless Drill" |
   | drill 2, other colour, scaled down via `<scale>` | OpenRobotics "Black and Decker Cordless Drill" |
   | mugs, large / small / other colour | GoogleResearch "Cole_Hardware_Mug_Classic_Blue", "Room_Essentials_Mug_White_Yellow", "ACE_Coffee_Mug_Kristen_16_oz_cup" |
   | tool distractor | GoogleResearch "Craftsman_Grip_Screwdriver_Phillips_Cushion" |
   | negative prompt "hammer" | GoogleResearch "Cole_Hardware_Hammer_Black", only placed for a counter-test |

   Later: NVIDIA SimReady Central (OpenUSD, needs a conversion per object to glTF/SDF).
2. Ground-truth boxes from a Gazebo `boundingbox_camera` sensor next to the colour camera, so
   IoU and "right instance" are computed automatically. Record ~200 frames at several distances
   and lighting conditions.
3. Prompt levels, the same for every object:

   | Level | Example | Tests |
   |---|---|---|
   | category | "drill" | basic recognition |
   | attribute | "green drill" | colour |
   | compound | "small green drill with black handle" | several attributes at once |
   | relational | "the drill next to the mug" | spatial understanding |
   | part | "handle of the drill" | needed for grasping |
   | negative | "hammer" (not in the scene) | false positives |

4. Report per model × prompt level: right-instance rate, mean IoU, false positives on negative
   prompts, latency mean / p95 on the 3080 Ti, VRAM. Expected: Grounding DINO / OWLv2 / YOLO-World
   latch onto the noun and pick the wrong drill on compound and relational prompts; VLMs are more
   accurate but 10–50x slower. This trade-off is the core result of the FM variant.
5. Closed-loop: servoing success rate and convergence time with detector periods
   0.2 s / 1 s / 5 s, with and without tracker.

### Own pipeline experiment (Elias, branch `feature/dino-sam2`)

Once `main` is clean (smooth tracking, `/gt/*` topics, FM image), branch off and try:
- **Grounding DINO + SAM 2 as tracker** (`tracker:=sam2`): DINO finds the object rarely from text,
  SAM 2 propagates the mask every frame, which also gives the orientation for yaw control.
- **SAM 3 / 3.1** as a single model that detects from text and tracks in streaming mode, as the
  counterpart: two specialised models vs. one general one, compared on speed and on the hard
  prompt levels above. The weights on Hugging Face are gated: request access early, it may take
  a while or be refused; the experiment is optional.
Merge into `main` only if it beats the current detector + CSRT pipeline.

#### Result 1: Grounding DINO + SAM 2 tracker (06.11.2026, sim, headless)

`tracker:=sam2` (SAM 2.1 tiny, Ultralytics `SAM2DynamicInteractivePredictor`): memory bank seeded
with the DINO box, refreshed every 15 frames from its own confident mask, lost when the object
score drops or the mask collapses/floods. `detect_on_loss:=true` calls DINO only until the first
track exists and after a loss. Start: `ros2 launch vs_bringup servo.launch.py tracker:=sam2
detect_on_loss:=true detector_period:=1.0`, then `/ibvs_node/start`.

| | DINO + SAM 2 | DINO + CSRT (1 s period) |
|---|---|---|
| Closed loop converged | yes, 30 s (3 runs, 30-42 s) | yes, 40 s |
| Final feature error norm | 0.004 | 0.003 |
| Camera over the red box | x 0.600, y 0.000, z 0.356 m | x 0.600, y 0.000, z 0.432 m |
| Features published (wall clock) | 4.1 Hz | 3.9 Hz |
| Features per simulated second | 12 | 19 |
| Detector calls | 2 in 30 s | 24 in 40 s |
| Cost per tracker frame | 160 ms idle GPU, ~240 ms with Gazebo | ~ms |
| Mask / orientation angle | yes | box only |

Measured at a real-time factor of 0.2-0.3 (Gazebo, DINO and the tracker share one laptop; DINO
took 0.9-1.4 s per call instead of the 0.31 s of an idle GPU; camera 4-7 Hz wall, 21 Hz sim),
so both columns are limited by the camera in wall-clock terms. SAM 2 does not raise the feature
rate over CSRT: it is launch-bound (about 150 ms per frame at 512 as well as 1024 px, bf16 does
not help), so the node feeds it only the newest frame. Its gain is robustness (detector runs
twice instead of 24 times) and the mask (centroid, area, angle). Not tested yet: occlusion and
the clutter world. The mask prompt path of Ultralytics 8.4.203 `update_memory` is broken, so
the memory refresh prompts with the mask's box.

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
