"""Moves the camera to the reference pose or a start configuration.

Uses the Gazebo ground truth of the target pose. This is part of the
experiment setup (defining start and goal), not of the control loop.

Can be used from Python, e.g. in an evaluation script:

    import rclpy
    from vs_sim.motion import RobotMover
    rclpy.init()
    mover = RobotMover()
    mover.go_to("aruco", start="S3_combined")   # or start=3, or start=None for the reference pose
    ...                                         # forward_velocity_controller is active now
"""

import math
import time

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import SwitchController
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from tf2_msgs.msg import TFMessage
from trajectory_msgs.msg import JointTrajectoryPoint

from vs_sim import targets as tg
from vs_sim.check_start_configs import Q_HOME
from vs_sim.kinematics import Chain, transform_msg_to_tf

TRAJ_CTRL = "scaled_joint_trajectory_controller"
VEL_CTRL = "forward_velocity_controller"


class RobotMover(Node):
    def __init__(self, name="vs_robot_mover", targets_file=None):
        super().__init__(name, parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.cfg = tg.load_config(targets_file)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._urdf = None
        self._joint_state = None
        self._model_poses = {}
        self.create_subscription(String, "/robot_description", self._on_urdf, latched)
        self.create_subscription(JointState, "/joint_states", self._on_js, 10)
        self.create_subscription(TFMessage, "/ground_truth/model_poses", self._on_poses, 10)
        self.target_pub = self.create_publisher(String, "/ground_truth/set_target", latched)
        self.vel_pub = self.create_publisher(Float64MultiArray, f"/{VEL_CTRL}/commands", 10)
        self.switch_cli = self.create_client(SwitchController, "/controller_manager/switch_controller")
        self.traj_cli = ActionClient(self, FollowJointTrajectory, f"/{TRAJ_CTRL}/follow_joint_trajectory")
        self.chain = None

    # ------------------------------------------------------------------ io
    def _on_urdf(self, msg):
        self._urdf = msg.data

    def _on_js(self, msg):
        self._joint_state = msg

    def _on_poses(self, msg):
        for t in msg.transforms:
            self._model_poses[t.child_frame_id] = transform_msg_to_tf(t.transform)

    def _spin_until(self, predicate, timeout, what):
        end = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > end:
                raise TimeoutError(f"Timeout while waiting for {what}")
            rclpy.spin_once(self, timeout_sec=0.05)

    def _wait_future(self, future, timeout, what):
        self._spin_until(future.done, timeout, what)
        return future.result()

    def sleep(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.02)

    def ready(self, timeout=30.0):
        self._spin_until(lambda: self._urdf is not None and self._joint_state is not None,
                         timeout, "/robot_description and /joint_states (is the simulation running?)")
        if self.chain is None:
            self.chain = Chain(self._urdf, "world", "camera_optical_frame")

    # ------------------------------------------------------------ poses
    def target_in_world(self, target, timeout=3.0):
        model = self.cfg["targets"][target]["model"]
        try:
            self._spin_until(lambda: model in self._model_poses, timeout, f"Gazebo pose of '{model}'")
            T_model = self._model_poses[model]
        except TimeoutError:
            self.get_logger().warn(f"No Gazebo pose for '{model}', using targets.yaml")
            T_model = tg.model_pose_from_config(self.cfg, target)
        return tg.target_frame(self.cfg, target, T_model)

    def resolve_start(self, start):
        """None -> reference pose; int (1-based) or name -> start configuration."""
        if start is None:
            return None, "reference"
        starts = tg.start_offsets(self.cfg)
        if isinstance(start, int) or (isinstance(start, str) and start.isdigit()):
            s = starts[int(start) - 1]
        else:
            matches = [s for s in starts if s["name"] == start]
            if not matches:
                raise ValueError(f"Unknown start configuration '{start}'")
            s = matches[0]
        return s["offset"], s["name"]

    def solve(self, target, start=None):
        """Joint configuration for target/start (deterministic seed, reproducible)."""
        self.ready()
        T_wt = self.target_in_world(target)
        q_ref, ok = self.chain.ik(tg.camera_goal_in_world(self.cfg, T_wt), Q_HOME, max_iter=3000)
        if not ok:
            raise RuntimeError(f"IK failed for the reference pose of '{target}'")
        offset, name = self.resolve_start(start)
        if offset is None:
            return q_ref, name
        q, ok = self.chain.ik(tg.camera_goal_in_world(self.cfg, T_wt, offset), q_ref, max_iter=3000)
        if not ok:
            raise RuntimeError(f"IK failed for '{target}' / '{name}'")
        return q, name

    # ------------------------------------------------------- controllers
    def switch(self, activate, deactivate, timeout=10.0):
        self._spin_until(lambda: self.switch_cli.service_is_ready(), timeout, "controller_manager")
        req = SwitchController.Request()
        req.activate_controllers = list(activate)
        req.deactivate_controllers = list(deactivate)
        req.strictness = SwitchController.Request.BEST_EFFORT
        req.activate_asap = True
        req.timeout = Duration(sec=5)
        res = self._wait_future(self.switch_cli.call_async(req), timeout, "switch_controller")
        if not res.ok:
            raise RuntimeError(f"Controller switch failed (activate={activate}, deactivate={deactivate})")

    def stop(self):
        """Zero joint velocities (velocity mode)."""
        self.vel_pub.publish(Float64MultiArray(data=[0.0] * 6))

    def velocity_mode(self):
        self.switch([VEL_CTRL], [TRAJ_CTRL])
        for _ in range(5):
            self.stop()
            self.sleep(0.02)

    def trajectory_mode(self):
        self.stop()
        self.switch([TRAJ_CTRL], [VEL_CTRL])

    def move_joints(self, q, max_joint_speed=0.6, timeout=60.0):
        self.ready()
        self._joint_state = None
        self._spin_until(lambda: self._joint_state is not None, 5.0, "/joint_states")
        q_now = self.chain.q_from_joint_state(self._joint_state)
        duration = max(2.0, float(np.max(np.abs(np.asarray(q) - q_now))) / max_joint_speed)
        self._spin_until(lambda: self.traj_cli.server_is_ready(), 10.0, f"{TRAJ_CTRL} action server")
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self.chain.joint_names)
        sec = int(math.floor(duration))
        goal.trajectory.points = [JointTrajectoryPoint(
            positions=[float(v) for v in q], velocities=[0.0] * len(q),
            time_from_start=Duration(sec=sec, nanosec=int((duration - sec) * 1e9)))]
        handle = self._wait_future(self.traj_cli.send_goal_async(goal), 10.0, "goal acceptance")
        if not handle.accepted:
            raise RuntimeError("Trajectory goal rejected")
        result = self._wait_future(handle.get_result_async(), timeout + duration, "trajectory execution")
        if result.result.error_code != 0:
            raise RuntimeError(f"Trajectory failed: {result.result.error_string}")

    # --------------------------------------------------------------- api
    def go_to(self, target, start=None, mode="velocity", settle=0.5):
        """Move to the reference pose (start=None) or a start configuration.

        mode: 'velocity'   -> forward_velocity_controller active afterwards
              'trajectory' -> scaled_joint_trajectory_controller stays active
        """
        if target not in self.cfg["targets"]:
            raise ValueError(f"Unknown target '{target}', available: {list(self.cfg['targets'])}")
        self.target_pub.publish(String(data=target))
        q, name = self.solve(target, start)
        self.get_logger().info(f"Moving to '{target}' / '{name}'")
        self.trajectory_mode()
        self.move_joints(q)
        self.sleep(settle)
        if mode == "velocity":
            self.velocity_mode()
        return q, name
