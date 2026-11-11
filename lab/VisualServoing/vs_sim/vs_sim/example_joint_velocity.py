"""Minimal interface example: rotates wrist_3 back and forth for a few seconds.

    ros2 run vs_sim go_to_start --target aruco --reference
    ros2 run vs_sim example_joint_velocity

Shows how to
  * read the camera image and camera info,
  * read the joint states and the camera pose (TF, robot kinematics),
  * command joint velocities via the forward_velocity_controller.

This is NOT a visual servoing controller - that is your task.
"""

import math

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float64MultiArray
from tf2_ros import Buffer, TransformListener

JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]  # order of the velocity command


class Example(Node):
    def __init__(self):
        super().__init__("example_joint_velocity", parameter_overrides=[Parameter("use_sim_time", value=True)])
        self.bridge = CvBridge()
        self.image = None
        self.K = None
        self.q = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(Image, "/camera/image_raw", self.on_image, 1)
        self.create_subscription(CameraInfo, "/camera/camera_info", self.on_info, 1)
        self.create_subscription(JointState, "/joint_states", self.on_js, 10)
        self.cmd = self.create_publisher(Float64MultiArray, "/forward_velocity_controller/commands", 10)
        self.t0 = None
        self.create_timer(0.02, self.step)  # 50 Hz control loop

    def on_image(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def on_info(self, msg):
        self.K = list(msg.k)

    def on_js(self, msg):
        pos = dict(zip(msg.name, msg.position))
        self.q = [pos[j] for j in JOINTS]

    def step(self):
        now = self.get_clock().now()
        if self.image is None or self.q is None or self.K is None:
            return
        if self.t0 is None:
            self.t0 = now
            h, w = self.image.shape[:2]
            self.get_logger().info(f"Image {w}x{h}, fx={self.K[0]:.1f} fy={self.K[4]:.1f} "
                                   f"cx={self.K[2]:.1f} cy={self.K[5]:.1f}")
            try:
                # T_tool0_camera: hand-eye transform from the URDF
                tr = self.tf_buffer.lookup_transform("tool0", "camera_optical_frame", Time())
                t = tr.transform.translation
                self.get_logger().info(f"camera_optical_frame in tool0: t=({t.x:.3f}, {t.y:.3f}, {t.z:.3f})")
            except Exception as e:  # noqa: BLE001
                self.get_logger().warn(f"TF not available yet: {e}")
        t = (now - self.t0).nanoseconds * 1e-9
        qd = [0.0] * 6
        if t < 6.0:
            qd[5] = 0.3 * math.sin(2.0 * math.pi * t / 3.0)  # wrist_3 [rad/s]
        self.cmd.publish(Float64MultiArray(data=qd))
        if t >= 6.0:
            self.get_logger().info("done")
            raise SystemExit


def main():
    rclpy.init()
    node = Example()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.cmd.publish(Float64MultiArray(data=[0.0] * 6))
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
