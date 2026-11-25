# Controller, gain study, small VLMs and box verifier

Results and design notes for the report and the slides (Philip's part). Numbers are from the
campaign `data/eval/final` (lab sim, UR5e, RTX 3050 Ti 4 GB) unless stated otherwise; every
table can be regenerated with the command given next to it.

## 1. Controller

**Control law.** Image-based visual servoing on decoupled region features of the target mask
(Tahri & Chaumette 2005):

| Feature | Definition | Couples mainly to |
|---|---|---|
| a_n | Z* · sqrt(a* / a), normalised area | v_z (depth) |
| x_n, y_n | a_n · x, a_n · y, centroid in normalised image coordinates | v_x, v_y |
| α | ½ atan2(2 μ11, μ20 − μ02), principal axis of the mask | ω_z (yaw) |

`v = −λ L⁺ (s − s*)` with the interaction matrix for (v_x, v_y, v_z, ω_z):

```
        x_n  [ -1   0   0   y_n ]
L =     y_n  [  0  -1   0  -x_n ]
        a_n  [  0   0  -1    0  ]
        α    [  0   0   0   -1  ]
```

With these features L is nearly constant, so the camera moves on an almost straight line.
ω_x and ω_y are not servoed from the image: forward kinematics keeps the optical axis vertical.
The angle error is wrapped modulo π (the principal axis has no direction); yaw is only servoed
when both the reference and the current mask have a defined axis (elongation > threshold), so
round objects (can, ball, cup) keep their yaw.

**From camera twist to joints.** Hand-eye transform from TF (`Ad(T_tool0←camera)`), rotated into
base axes, then damped least squares `q̇ = Jᵀ (J Jᵀ + d² I)⁻¹ v` (d = 0.05) with the
`LOCAL_WORLD_ALIGNED` Jacobian from Pinocchio. Limits: 0.1 m/s and 0.5 rad/s on the camera twist,
50 % of the joint velocity limits (scaled uniformly, direction kept), no motion further into a
joint limit within 0.1 rad.

**Safety.** The arm holds (zero joint velocities) when features are stale (> 0.5 s), when the
look-ahead of 0.2 s predicts self-collision, a frame below 3 cm or tool0 leaving the workspace box.
Motion out of a violation stays allowed. A lost target (perception reports it, or nothing for
2 s) ends servoing instead of holding, so a later detection of another object cannot pull the arm
away. Zero command on shutdown (the lab's velocity controller keeps the last command).
Safe-stop test (`lost` block): arm still after ~0.5 s, ~5 mm drift.

**Measured behaviour.** The error decays at ~0.9 λ (exponential fit on the ground-truth error,
one smoke run on red_can S1 per λ from 0.3 to 1.5) with ~0.1 s dead time between the first
command and the first camera motion.

**Known limitations** (from the review):
- Linear and angular velocity are clamped separately: in saturation the 6D direction changes
  and the straight-line property is lost.
- The stop criterion ‖e‖ < 0.01 mixes metres (x_n, y_n, a_n) and radians (α).
- Latency compensation is off by default (`latency_compensation: 0.0`): features up to 0.5 s
  old are used as they are, which is where the overshoot at high λ comes from. Section 2a
  measures the optional first-order prediction.
- 180° ambiguity of symmetric shapes (start S7 of blue_box).

## 2. Gain study (λ vs convergence time and overshoot)

`gain_*` blocks: λ ∈ {0.3, 0.6, 1.0, 1.5, 2.5} on red_can (no yaw) and blue_box (with yaw),
8 start configurations each, Grounding DINO + SAM 2 (λ = 0.6 is `main_dino_sam2`).
`$VS_PYTHON -m vs_evaluation.gain_study --campaign /root/vs_data/eval/final`, notebook section 6b.

| λ | 0.3 | 0.6 | 1.0 | 1.5 | 2.5 |
|---|---|---|---|---|---|
| Convergence time, median, red_can [s] | 9.0 | 4.9 | 3.9 | 2.6 | 1.5 |
| Convergence time, median, blue_box [s] | 14.7 | 7.9 | 5.5 | 4.0 | 2.0 |
| Largest overshoot, red_can [mm] | 0.0 | 0.6 | 2.6 | 3.7 | 6.3 |
| Largest overshoot, blue_box [mm] | 0.1 | 1.7 | 6.7 | 10.9 | 19.3 |
| Final position error, median, red_can [mm] | 6.3 | 4.3 | 2.5 | 1.1 | 0.9 |
| Peak camera speed, median, red_can [m/s] | 0.017 | 0.035 | 0.058 | 0.087 | 0.100 |

Convergence time is measured until the error first stays below the threshold (the 1 s hold is
not included). Overshoot: distance past the reference pose per axis against the sign of the
initial error, axes starting below 5 mm skipped.

**Reading.**
- Convergence time falls roughly as 1/λ up to λ ≈ 1.5. From there the speed limit of 0.1 m/s
  caps the peak speed, so a larger λ hardly helps but the overshoot keeps growing.
- The overshoot is mostly along y and largest on start S6 of blue_box (19 mm at λ = 2.5); on
  red_can it stays below 7 mm.
- λ = 0.6–1.0 is the sensible range: no or a few mm of overshoot, 4–8 s.
- **The final error depends on λ through the stop criterion, not through the accuracy of the
  method:** a run ends once the feature error has stayed below 0.01 for 1 s. At λ = 0.3 the
  camera is still ~6 mm away at that point and creeps closer only slowly; at λ ≥ 1.5 it has
  moved further during the hold. A fixed 5 mm tolerance therefore turns λ = 0.3 into 0/8 on
  red_can although every run converged. This is why the evaluation reports error distributions
  (as agreed with the prof) rather than one success rate.

### 2a. Latency compensation

The controller acts on features that are ~0.1 s old (dead time measured above, plus the age of
the last feature message). With `latency_compensation: Δt > 0` (ibvs_node parameter) it first
predicts them Δt + age ahead with its last commanded camera twist, `ŝ = s + Δt L(s) v`
(`ibvs.predict_features`, first order, a Smith-predictor-like correction), and runs the control
law on ŝ. Zero after a halt, so nothing is predicted from a stale command.
`gainlc_*` blocks: λ ∈ {1.0, 1.5, 2.5} with Δt = 0.1 s, otherwise identical to `gain_*`
(`run_eval --ibvs-param latency_compensation=0.1`, restored after the block).

| λ | 1.0 | 1.0 + LC | 1.5 | 1.5 + LC | 2.5 | 2.5 + LC |
|---|---|---|---|---|---|---|
| Largest overshoot, blue_box [mm] | 6.7 | 3.0 | 10.9 | 3.6 | 19.3 | 5.0 |
| Largest overshoot, red_can [mm] | 2.6 | 2.0 | 3.7 | 3.0 | 6.3 | 3.8 |
| Convergence time, median, blue_box [s] | 5.5 | 5.2 | 4.0 | 3.4 | 2.0 | 2.7 |
| Convergence time, median, red_can [s] | 3.9 | 3.7 | 2.6 | 2.1 | 1.5 | 1.4 |
| Success (5 mm / 2°), blue_box | 8/8 | 8/8 | 8/8 | 8/8 | 6/8 | 8/8 |
| Success (5 mm / 2°), red_can | 7/8 | 6/8 | 6/8 | 7/8 | 5/8 | 6/8 |

**Reading.**
- The worst-case overshoot drops by a factor of 2–4 on blue_box and by 18–39 % on red_can,
  without slowing convergence (except blue_box at λ = 2.5, +0.7 s, where the speed limit is
  active anyway). With compensation λ = 1.5 overshoots no more than λ = 1.0 without it.
- At λ = 2.5 the two blue_box runs that failed without compensation succeed with it.
- 8 runs per cell: the overshoot maxima are clear, the success differences of ±1 run are not.
- The final error at λ = 1.0 on red_can gets larger (median 2.5 → 4.8 mm): with less overshoot
  the camera approaches from one side and stops at the threshold, the stop-criterion effect
  again (section 2b).
- Recommendation: λ = 1.5 with `latency_compensation: 0.1` for speed; the default stays
  λ = 0.6 without compensation, which is what all other blocks were run with.

### 2b. Error metrics independent of the stop criterion

`success` mostly measures where the controller stops (≈ 4–5 mm at λ = 0.6, see above).
`vs_evaluation.error_metrics` takes the ground-truth error over time instead: share of runs that
ever got below 10 / 5 / 2 mm and the median time to get there, smallest error reached, and the
integral of the error over the first 15 s (IAE, after a converged run its last error is held).
`$VS_PYTHON -m vs_evaluation.error_metrics --campaign /root/vs_data/eval/final`, notebook 6c.

| red_can | λ 0.6 | λ 1.0 | λ 1.5 | λ 2.5 | λ 1.5 + LC | λ 2.5 + LC |
|---|---|---|---|---|---|---|
| Reached 5 mm | 8/8 | 7/8 | 6/8 | 5/8 | 7/8 | 6/8 |
| Reached 2 mm | 0/8 | 3/8 | 5/8 | 5/8 | 4/8 | 5/8 |
| Smallest error, median [mm] | 4.3 | 2.6 | 1.2 | 0.7 | 2.2 | 1.1 |
| IAE over 15 s, median [mm·s] | 205 | 151 | 141 | 136 | 158 | 139 |

At λ = 0.6 no red_can run ever gets below 2 mm: the method could, but the controller stops
first. A smaller `converged_threshold` (or one in mm instead of mixed units) would lower the final
error at every λ at the cost of time; the IAE shows that above λ ≈ 1.5 the gain no longer buys
much (136–141 mm·s).

## 3. Small VLMs on a 4 GB GPU

Qwen3-VL-4B (as `qwen3-vl-gpu:4b`: all 37 layers forced onto the GPU,
`scripts/ollama/qwen3-vl-gpu-4b.Modelfile`; Ollama otherwise put half of them on the CPU, 9 s
instead of ~1.4 s per frame) and Gemma 4 E2B, both 4-bit via Ollama, as detectors that seed SAM 2.

**Closed loop** (`det_qwen3vl_4b`, `det_gemma4_e2b`, 4 targets × 8 starts):

| Target | Grounding DINO | Gemma 4 E2B | Qwen3-VL-4B |
|---|---|---|---|
| red_can | 8/8 | 3/8 | 0/8 (not recognised) |
| blue_box | 8/8 | 8/8 | 8/8 |
| green_cup | 5/8 | 1/8 (median 11 mm off) | 0/8 (not recognised) |
| yellow_ball | 4/8 | 2/8 | 0/8 (not recognised) |

**Offline benchmark** (`bench_detectors`, 31 frames of `main_dino_sam2`, share of correct boxes
IoU ≥ 0.5 per prompt level, false positives on prompts of absent objects):

| Detector | attribute | category | synonym | description | FP absent | Latency |
|---|---|---|---|---|---|---|
| Grounding DINO | 1.00 | 0.61 | 1.00 | 0.51 | 1.00 | 0.55 s |
| YOLO-World | 0.52 | 0.17 | 0.28 | 0.04 | 0.00 | 0.02 s |
| OWLv2 | 0.48 | 0.27 | 0.26 | 0.29 | 0.00 | 0.52 s |
| Qwen3-VL-4B | 0.70 | 0.57 | 0.56 | 0.59 | 0.00 | 1.16 s |
| Gemma 4 E2B | 0.96 | 0.33 | 0.95 | 0.42 | 0.10 | 1.71 s |
| Gemma 4 E2B, thinking off | 0.95 | 0.29 | 0.88 | 0.46 | 0.06 | 2.26 s* |

\* separate run right after a campaign on a warm GPU; with the object in view both modes take
~2 s in a direct back-to-back test, the difference is in the absent-object case (see below).

**Reading.**
- Qwen3-VL-4B does not call the can seen from above a "can" (nor a "red cylinder"); asked for a
  "red object" it returns the right box. It never claims an absent object, but in the closed
  loop it only finds the box, so it fails at the reference pose of the other three targets.
- Gemma 4 E2B finds every object but its boxes are looser; SAM 2 seeded from them converges a
  few mm off (green_cup 11 mm median).
- Gemma 4 decides itself whether to think: with the object in view it answers directly (~2 s),
  for an absent object it reasons first (4.2 s median, up to 15 s). With thinking off
  (`vlm_think: false`, block `det_gemma4_e2b_nothink`) the answer time stays bounded (absent:
  1.2 s median, at most 2.4 s), false positives on absent objects drop from 0.10 to 0.06,
  attribute accuracy is unchanged (0.96 / 0.95). Closed loop: 16/32 vs 14/32 successful,
  mean convergence 7.0 s vs 8.0 s, within the run-to-run spread of a VLM.
- On 4 GB both are usable as detectors seeding a tracker, not as a per-frame localiser.

## 4. SigLIP 2 box verifier

Grounding DINO always returns a box, also for objects that are not in the scene (FP on absent
objects 1.00 above; the ArUco board tracked as "red can" earlier). The verifier
(`vs_perception/verifiers.py`, `verifier:=siglip`) crops each detector box (square, 15 % context)
and scores it against "a photo of a <prompt>." with SigLIP 2 (`google/siglip2-base-patch16-224`).
SigLIP's sigmoid loss gives an absolute match score per image-text pair, so no list of
alternatives is needed (CLIP's softmax would need one). Boxes below 0.002 are dropped before the
tracker is seeded.

**Calibration** (`vs_evaluation.bench_verifier`, ground-truth crops of the 31 frames): at 0.002,
99 % of the attribute prompts are kept, 3 % of other objects' prompts and 0 % of absent objects'
prompts pass. Bare categories ("can") score much lower (57 % kept): meant for descriptive prompts.

**Effect** (`bench_detectors --verifier siglip`, same detections with and without):

| Detector | FP absent | attribute correct | Extra time |
|---|---|---|---|
| Grounding DINO | 1.00 → **0.00** | 1.00 → 0.99 | +21 ms |
| Gemma 4 E2B | 0.10 → **0.00** | 0.96 → 0.92* | +58 ms |
| YOLO-World, OWLv2 | 0.00 → 0.00 | unchanged | +2–8 ms |

\* raw row from an earlier run, so part of the difference is VLM run-to-run variation.

**Closed loop:** see section 5.

## 5. Closed-loop check of the verifier

Grounding DINO + SAM 2, arm at red_can start S1 (can, box, cup and ball in view), prompt set,
6 s for the detector, then `~/start` and 10 s of servoing (probe script, not part of the repo).

| Prompt | Verifier none (main configuration) | Verifier siglip |
|---|---|---|
| "purple bottle" (absent) | 100 % valid features, **camera moved 94.6 mm** | 0 % valid, **camera did not move** |
| "black phone" (absent) | 100 % valid features, **camera moved 91.7 mm** | 0 % valid, **camera did not move** |
| "red can" (present) | 100 % valid, servoing | 100 % valid, servoing |

Without the verifier the arm drives towards whatever object Grounding DINO picked for a prompt
that matches nothing; with it, no box survives and the controller never gets a target.

**Normal operation** (`det_dino_siglip` vs `main_dino_sam2`, 4 targets × 8 starts):

| | Verifier none | Verifier siglip |
|---|---|---|
| Converged | 32/32 | 32/32 |
| Within 5 mm / 2° | 25/32 | 28/32 |
| Convergence time, median (without the 1 s hold) | 5.7 s | 5.0 s |
| Final position error, median | 4.28 mm | 4.24 mm |

The verifier costs nothing in normal operation. The three extra successes are runs that end
just above or below the 5 mm tolerance (see section 2), not an effect of the verifier.
(`main_dino_sam2` was recorded before run_eval subtracted the hold time; its 6.7 s raw median
is 5.7 s on the same definition.)

## Sources

- O. Tahri, F. Chaumette: "Point-based and region-based image moments for visual servoing of
  planar objects", IEEE T-RO 21(6), 2005.
- F. Chaumette, S. Hutchinson: "Visual servo control, Part I: Basic approaches", IEEE RAM 13(4),
  2006.
- X. Zhai et al.: "Sigmoid Loss for Language Image Pre-Training" (SigLIP), ICCV 2023;
  M. Tschannen et al.: "SigLIP 2", 2025. Weights `google/siglip2-base-patch16-224`, Apache 2.0.
- Qwen3-VL (Alibaba, Apache 2.0), Gemma 4 (Google, Apache 2.0), served by Ollama (MIT).
- Pinocchio (LAAS-CNRS, BSD-2) for kinematics, Jacobian and collision checking.
