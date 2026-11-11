"""Ground truth for the evaluation (NOT allowed for the control loop!).

Subscribes:
    /ground_truth/model_poses   tf2_msgs/TFMessage   all model poses from Gazebo (bridged)
    /tf, /tf_static                                   camera pose from the robot kinematics
    /ground_truth/set_target    std_msgs/String       switch the active target

Publishes (all in frame 'world' unless noted), 30 Hz:
    /ground_truth/target_pose          geometry_msgs/PoseStamped  target frame (top surface of the object)
    /ground_truth/camera_pose          geometry_msgs/PoseStamped  camera_optical_frame
    /ground_truth/desired_camera_pose  geometry_msgs/PoseStamped  camera_optical_frame in the reference pose
    /ground_truth/camera_in_target     geometry_msgs/PoseStamped  camera pose in the target frame (frame: target)
    /ground_truth/pose_error           std_msgs/Float64MultiArray
        [ex, ey, ez, rx, ry, rz, |e_t|, |e_r|]
        translation [m] and rotation vector [rad] of the current camera pose
        w.r.t. the desired camera pose, expressed in the desired camera frame.

Parameter:
    active_target (string, default 'aruco')
"""

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import Float64MultiArray, String
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener

from vs_sim import targets as tg
from vs_sim.kinematics import (inv_tf, pose_error, tf_to_pose_msg,
                               transform_msg_to_tf)

CAMERA_FRAME = "camera_optical_frame"


class GroundTruth(Node):
    def __init__(self):
        super().__init__("ground_truth")
        self.declare_parameter("active_target", "aruco")
        self.declare_parameter("targets_file", "")
        path = self.get_parameter("targets_file").value or None
        self.cfg = tg.load_config(path)
        self.model_poses = {}
        self.warned_fallback = set()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(TFMessage, "/ground_truth/model_poses", self.on_poses, 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/ground_truth/set_target", self.on_set_target, latched)

        self.pub_target = self.create_publisher(PoseStamped, "/ground_truth/target_pose", 10)
        self.pub_cam = self.create_publisher(PoseStamped, "/ground_truth/camera_pose", 10)
        self.pub_des = self.create_publisher(PoseStamped, "/ground_truth/desired_camera_pose", 10)
        self.pub_cit = self.create_publisher(PoseStamped, "/ground_truth/camera_in_target", 10)
        self.pub_err = self.create_publisher(Float64MultiArray, "/ground_truth/pose_error", 10)
        self.create_timer(1.0 / 30.0, self.tick)
        self.get_logger().info(f"Ground truth running, active target: {self.active_target}")

    @property
    def active_target(self):
        return self.get_parameter("active_target").value

    def on_set_target(self, msg):
        if msg.data not in self.cfg["targets"]:
            self.get_logger().error(f"Unknown target '{msg.data}'")
            return
        self.set_parameters([Parameter("active_target", value=msg.data)])
        self.get_logger().info(f"Active target: {msg.data}")

    def on_poses(self, msg):
        for t in msg.transforms:
            self.model_poses[t.child_frame_id] = transform_msg_to_tf(t.transform)

    def target_in_world(self, name):
        model = self.cfg["targets"][name]["model"]
        if model in self.model_poses:
            T_model = self.model_poses[model]
        else:
            if model not in self.warned_fallback:
                self.get_logger().warn(f"No Gazebo pose for '{model}' yet, using targets.yaml")
                self.warned_fallback.add(model)
            T_model = tg.model_pose_from_config(self.cfg, name)
        return tg.target_frame(self.cfg, name, T_model)

    def _msg(self, T, frame, stamp):
        m = PoseStamped()
        m.header.frame_id = frame
        m.header.stamp = stamp
        tf_to_pose_msg(T, m.pose)
        return m

    def tick(self):
        name = self.active_target
        if name not in self.cfg["targets"]:
            return
        stamp = self.get_clock().now().to_msg()
        T_wt = self.target_in_world(name)
        T_wd = tg.camera_goal_in_world(self.cfg, T_wt)
        self.pub_target.publish(self._msg(T_wt, "world", stamp))
        self.pub_des.publish(self._msg(T_wd, "world", stamp))
        try:
            tr = self.tf_buffer.lookup_transform("world", CAMERA_FRAME, Time())
        except Exception:
            return
        T_wc = transform_msg_to_tf(tr.transform)
        stamp = tr.header.stamp
        self.pub_cam.publish(self._msg(T_wc, "world", stamp))
        self.pub_cit.publish(self._msg(inv_tf(T_wt) @ T_wc, "target", stamp))
        e_t, e_r = pose_error(T_wc, T_wd)
        self.pub_err.publish(Float64MultiArray(
            data=[float(v) for v in np.concatenate([e_t, e_r, [np.linalg.norm(e_t), np.linalg.norm(e_r)]])]))


def main():
    rclpy.init()
    node = GroundTruth()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
