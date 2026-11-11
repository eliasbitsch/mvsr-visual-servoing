# MVSR Lab – Visual Servoing with Foundation Models

Team: Elias, Philip, Jiayi · Variant: **FM** · Spec: `moodle/4_Projektangaben/1_Angabe Visual Servoing.pdf`

## Goal in one sentence

An eye-in-hand camera on the UR5e of the lab simulation finds an object described in natural
language ("the red can") with a foundation model, and an image-based visual servoing (IBVS) loop
drives the camera into the reference pose 0.30 m above that object, using only image information
for control.

## Simulation

Since 11.11.2026 we use the lecturer's lab simulation unchanged (repo root: `Dockerfile`,
`docker-compose.yml`, `vs_sim/`, documented in [LAB_SIM.md](LAB_SIM.md)). The earlier GoFa cell is
gone. FM targets: `red_can`, `blue_box`, `green_cup`, `yellow_ball`; eight start configurations per
target via `go_to_start`; ground truth on `/ground_truth/*` for the evaluation only.

## Status (11.11.2026)

Working in the lab simulation, measured against `/ground_truth/pose_error`:

| Run | Converged | Position error | Rotation error |
|---|---|---|---|
| red_can S1, DINO + CSRT | 6.6 s | 1.6 mm | 0.02 deg |
| blue_box S3 (-35 deg), DINO + SAM 2, yaw servoed | 8.1 s | 1.4 mm | 0.05 deg |
| red_can S3, DINO + SAM 2 | 8.3 s | 3.0 mm | 35 deg, round from above, no yaw observable |
| blue_box S7 (170 deg), DINO + SAM 2 | 11.9 s | 0.4 mm | 180 deg, symmetric shape, view identical to reference |

- Workflow: `go_to_start --reference`, `capture_reference`, `go_to_start --start N`, `start`.
- Web GUI (Lichtblick, port 8081) with 3D robot and objects, camera, live plots, service buttons;
  the simulation runs headless.
- Load: about 7.3 of 20 threads headless (our nodes ignore the ~500 Hz `/clock`, control 30 Hz,
  camera 15 Hz). The Gazebo window costs another ~1.4 cores.
- Safety: stopping the stack sends zero joint velocities (the lab's velocity controller keeps the
  last command otherwise).
- Open findings: yaw needs a mask with a defined axis (SAM 2); with the can out of view DINO tracked
  the ArUco board as "red can" (false positive, needs a plausibility check).

## Architecture

```
            Lab simulation (vs_sim, image vs-sim:jazzy)               ROS 2 Jazzy
 ┌──────────────────────────────────────────────┐
 │ UR5e + wrist camera (RGB, 640x480)           │
 │ ros2_control: joint_state_broadcaster,       │
 │   forward_velocity_controller,               │
 │   scaled_joint_trajectory_controller         │
 └──────┬──────────────────────────▲────────────┘
        │ /camera/image_raw          │ /forward_velocity_controller/commands
        ▼                            │
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
 evaluation (to do): go_to_start for each start config, /ground_truth/pose_error only for metrics
```

Our ROS packages in `student_ws/src` (image `vs-fm:jazzy` from `docker/Dockerfile.fm`):

| Package | Content | Owner |
|---|---|---|
| `vs_bringup` | Launch file and parameters for the lab simulation | Elias |
| `vs_interfaces` | `ObjectFeatures.msg` | – |
| `vs_perception` | FM plug-in framework, trackers, feature extraction, perception node | Elias (core), everyone (adapters) |
| `vs_control` | IBVS node, hand-eye transform, differential kinematics, safety | Philip |
| `vs_evaluation` | Evaluation runner, metrics, plots | Jiayi |

## Foundation model assignment

Every FM is one file in `vs_perception/vs_perception/detectors/`. All three of us implement the
same `Detector` interface, so the node, the tracker and the controller never change.

| Person | GPU | Foundation models | Why this split |
|---|---|---|---|
| **Elias** | RTX 3080 Ti Laptop, 16 GB | **Grounding DINO + SAM 2** (main pipeline) + **larger VLMs** (Qwen3-VL-8B, Gemma 4 E4B, GLM-4.6V) | Heaviest models, needs the VRAM; also runs the live demo |
| **Philip** | RTX 3050, ~4 GB | **small VLMs** (Qwen3-VL-4B, Gemma 4 E2B, 4-bit via Ollama) + **YOLO-World** as fast baseline | VLMs that fit 4 GB; together with Elias' VLMs this gives the size/latency curve |
| **Jiayi** | weak GPU / CPU | **OWLv2** + **Florence-2** (classic open-vocabulary detectors) | Small enough for CPU; evaluated offline on the recorded image set, so no live sim needed on her laptop |

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

1. Recorded in the lab world (the evaluation world is fixed): camera images of the four FM
   targets from all eight start configurations and along servoing runs (`go_to_start`, then
   save `/camera/image_raw`). Ground-truth boxes come from projecting `/ground_truth/model_poses`
   into the camera with the known intrinsics, so IoU and "right instance" are computed
   automatically.
2. Optional extra, outside the graded evaluation: a copy of the lab world with distractors that
   only the prompt can separate (Gazebo Fuel objects, or NVIDIA SimReady assets converted from
   OpenUSD), to test compound and relational prompts.
3. Prompt levels, the same for every object:

   | Level | Example | Tests |
   |---|---|---|
   | category | "can" | basic recognition |
   | attribute | "red can" | colour |
   | compound | "small red cylindrical can" | several attributes at once |
   | relational | "the can next to the blue box" | spatial understanding |
   | part | "top of the can" | needed for grasping |
   | negative | "hammer" (not in the scene) | false positives |

4. Report per model × prompt level: right-instance rate, mean IoU, false positives on negative
   prompts, latency mean / p95 on the 3080 Ti, VRAM. Expected: Grounding DINO / OWLv2 / YOLO-World
   latch onto the noun and pick the wrong object on compound and relational prompts; VLMs are more
   accurate but 10–50x slower. This trade-off is the core result of the FM variant.
5. Closed-loop: servoing success rate and convergence time with detector periods
   0.2 s / 1 s / 5 s, with and without tracker.

### Evaluation metrics (spec section 2.5)

The runner (to do, Jiayi with Elias) drives every target x start configuration automatically and
uses the ground truth only for the numbers. Proposed definitions, to confirm in the team:

| Metric | Definition |
|---|---|
| Success | converged within 60 s and final camera pose within 5 mm and 2 deg of the reference pose; for round objects (can, ball) the yaw is excluded, for rectangles taken modulo 180 deg |
| Final pose error | translation and rotation of `/ground_truth/pose_error` at the end of the run |
| Final feature error | norm of `/ibvs_node/error` at the end |
| Convergence time | from `start` until the feature error stays below the threshold for 1 s |
| Time series | feature error and commanded camera velocity over time (`/ibvs_node/error`, `/ibvs_node/camera_twist`) |
| Trajectories | camera path in Cartesian space (`/ground_truth/camera_pose`) and feature path in the image (`/vs/features`) |

Critical analysis required by the spec: which start configurations converge reliably, robustness
to partial occlusion, lighting changes and larger distances (S4), failure cases (target lost from
view, joint limits, singularities, S7), and the limits of the approach (symmetric objects, false
positives, FM latency).

### Own pipeline experiment (Elias)

Merged into `main` on 11.11.2026; still to repeat in the lab simulation:
- **Grounding DINO + SAM 2 as tracker** (`tracker:=sam2`): DINO finds the object rarely from text,
  SAM 2 propagates the mask every frame, which also gives the orientation for yaw control.
- **SAM 3 / 3.1** as a single model that detects from text and tracks in streaming mode, as the
  counterpart: two specialised models vs. one general one, compared on speed and on the hard
  prompt levels above. The weights on Hugging Face are gated: request access early, it may take
  a while or be refused; the experiment is optional.
Merge into `main` only if it beats the current detector + CSRT pipeline.

#### Result 1: Grounding DINO + SAM 2 tracker (06.11.2026, earlier GoFa cell, headless)

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
safe stop). Philip owns them together with the controller.

### Optional, only once servoing and evaluation are done

An LLM agent on top ("pick the red can and put it next to the blue box"): a language model plans
with a small set of skills (`servo_to(prompt)`, `open_gripper`, `close_gripper`,
`move_to_named_pose`), each skill calls the existing nodes, and the safety layer in
`vs_control` still has the last word. Beyond the assignment; worth it only as an extra in the
demo.

## Milestones

| When | Done means |
|---|---|
| Online session 1 (11.11.) | Lab sim runs, camera image visible, robot moves. Done, plus the closed loop on red_can. Open: each person's FM on saved sim images. |
| Online session 2 (17.11.) | Closed loop with Grounding DINO + tracker for x/y/z/yaw from one start pose (done). Other FMs plugged into the node (Florence-2, small VLMs missing). Evaluation runner runs one trial automatically. |
| Online session 3 (01.12.) | All start configurations evaluated, FM comparison table, latency study, failure cases collected. Slides draft. |
| 09.12. 23:59 (presentation 10.12.) | Zip submitted (code, README, slides, demo video), clean install tested on a fresh machine from the README alone. |

## How to add a foundation model

1. Create `vs_perception/vs_perception/detectors/my_model.py`, subclass `Detector`, decorate with
   `@register_detector("my_model")`, implement `load()` and `detect(image_rgb, prompt)`.
2. Import it in `detectors/__init__.py`.
3. Run `ros2 launch vs_bringup servo.launch.py detector:=my_model prompt:="red can"`.

Details on the interface: `student_ws/src/vs_perception/README.md`.
