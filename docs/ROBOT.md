# Robot reference: GoFa cell

Everything needed to operate the simulated cell by hand or from code. Values come from the
URDF (`gofa_description`, `vs_bringup/urdf/vs_cell.urdf.xacro`) and the controller config.

## Hardware in the cell

| Part | Model | Mounted on |
|---|---|---|
| Arm | ABB GoFa CRB 15000-5/0.95, 6 revolute joints, 5 kg payload, 950 mm reach | `world` (fixed) |
| Gripper | Robotiq 2F-85, parallel, 85 mm stroke | `tool0` |
| Camera | Intel RealSense D435, colour 640x480 @ 30 Hz + aligned depth | `tool0`, beside the gripper, looking along tool z |

## Joints

| Joint | Axis | Limits [deg] | Max speed [deg/s] | Start pose [rad] |
|---|---|---|---|---|
| `joint_1` | base yaw (z) | −180 … 180 | 125 | 0 |
| `joint_2` | shoulder (y) | −180 … 180 | 125 | 0 |
| `joint_3` | elbow (y) | −225 … 85 | 140 | 0 |
| `joint_4` | forearm roll (x) | −180 … 180 | 200 | 0 |
| `joint_5` | wrist pitch (y) | −180 … 180 | 200 | 1.5708 |
| `joint_6` | flange roll (x) | −180 … 180 | 200 | 0 |
| `robotiq_85_left_knuckle_joint` | gripper | 0 (open) … 0.8 (closed) | | 0 |

In the start pose the forearm is horizontal and the camera looks straight down, about 0.7 m
above the floor at x ≈ 0.55 m. The visual servoing controller uses only 50 % of the max speeds
(`joint_speed_scale`) and stays 0.1 rad away from every limit (`joint_limit_margin`).

## Frames

`world` = `base_link` → `link_1` … `link_6` → `flange` → `tool0` (z out of the flange)
→ gripper `robotiq_85_base_link`, camera `camera_link` → `camera_color_optical_frame`
(z = viewing direction, x right, y down in the image). The camera pose relative to `tool0` is
the hand-eye transform; code reads it from TF and never hard-codes it.

## Controllers

| Controller | Type | Active at start | Command topic |
|---|---|---|---|
| `joint_state_broadcaster` | state | yes | – |
| `arm_trajectory_controller` | joint trajectory (position) | yes | `/arm_trajectory_controller/joint_trajectory` |
| `arm_velocity_controller` | joint group velocity | no | `/arm_velocity_controller/commands` |
| `gripper_controller` | joint group position | yes | `/gripper_controller/commands` |

Only one of the two arm controllers can be active. `/ibvs_node/start` and `/ibvs_node/stop`
switch between them.

## Commands

```bash
# Move to a joint configuration (here: the start pose) in 3 s
ros2 topic pub --once /arm_trajectory_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory \
  "{joint_names: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6],
    points: [{positions: [0.0, 0.0, 0.0, 0.0, 1.5708, 0.0], time_from_start: {sec: 3}}]}"

# Gripper: 0.0 open, 0.8 closed
ros2 topic pub --once /gripper_controller/commands std_msgs/msg/Float64MultiArray "{data: [0.8]}"

# Visual servoing
ros2 service call /ibvs_node/start std_srvs/srv/Trigger
ros2 service call /ibvs_node/stop std_srvs/srv/Trigger
ros2 service call /ibvs_node/capture_reference std_srvs/srv/Trigger

# Change the target or the detector rate while running
ros2 param set /perception_node prompt "blue cylinder"
ros2 param set /perception_node detector_period 0.5

# Inspect
ros2 control list_controllers
ros2 topic echo /ibvs_node/error
ros2 run tf2_ros tf2_echo tool0 camera_color_optical_frame
```

## Safety rules

Enforced in code (`vs_control`), not by convention. A prompt or a comment cannot guarantee
any of this; the controller refuses commands that break a rule.

| Rule | Where | Behaviour |
|---|---|---|
| Stop when the target is lost, invalid or features are older than 0.5 s | `ibvs_node` | zero joint velocities until the target is back |
| Cartesian speed ≤ 0.10 m/s, rotation ≤ 0.5 rad/s | `ibvs_node` | twist scaled down, direction kept |
| Joint speed ≤ 50 % of the URDF limit | `kinematics.limit_joint_velocities` | whole vector scaled, direction kept |
| Never closer than 0.1 rad to a joint limit | `kinematics.limit_joint_velocities` | motion into the limit blocked |
| No self-collision (arm, gripper, camera; 1 cm margin) | `safety.SafetyMonitor` | stop if a collision is predicted within 0.2 s; backing out allowed |
| Tool, camera and finger tips stay ≥ 3 cm above the floor | `safety.SafetyMonitor` | downward motion blocked |
| `tool0` stays inside x 0.15–1.0, y −0.7–0.7, z 0–1.2 m | `safety.SafetyMonitor` | motion leaving the box blocked |
| Singularities | `kinematics.damped_pinv_solve` | damped least squares keeps joint speeds bounded |
| Ground truth `/gt/*` only for evaluation | convention, checked in review | the controller never subscribes to it |

Grasping is not part of the assignment. If it is added: the gripper stays closed while the arm
moves with an object and opens only in the place pose.
