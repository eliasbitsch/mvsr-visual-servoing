"""Closed-loop IBVS: image features -> camera twist -> hand-eye -> joint velocities.

Safety: the arm stops (zero joint velocities) whenever features are missing, invalid or older
than feature_timeout, and whenever SafetyMonitor predicts self-collision or leaving the
workspace. The ground-truth topics /gt/* are never used here.
"""
import numpy as np
import pinocchio as pin
import rclpy
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import TwistStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Bool, Float64MultiArray, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from vs_interfaces.msg import ObjectFeatures

from vs_control import ibvs
from vs_control.kinematics import (adjoint, clamp_norm, damped_pinv_solve,
                                   limit_joint_velocities, transform_to_rp)
from vs_control.safety import SafetyMonitor

PARAMS = {
    'camera_info_topic': '/camera/camera/color/camera_info',
    'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
    'base_frame': 'base_link', 'ee_frame': 'tool0', 'camera_frame': 'camera_color_optical_frame',
    'arm_joints': ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'],
    'command_topic': '/arm_velocity_controller/commands',
    'velocity_controller': 'arm_velocity_controller',
    'position_controller': 'arm_trajectory_controller',
    'rate': 50.0, 'gain': 0.6,
    'desired_u': -1.0, 'desired_v': -1.0, 'desired_area': 9000.0, 'desired_angle': 0.0,
    'desired_depth': 0.35, 'control_yaw': False,
    'align_optical_axis_down': True, 'align_gain': 1.0,
    'max_linear_speed': 0.10, 'max_angular_speed': 0.5, 'joint_speed_scale': 0.5,
    'joint_limit_margin': 0.1, 'dls_damping': 0.05, 'feature_timeout': 0.5,
    'converged_threshold': 0.01, 'converged_hold_time': 1.0, 'auto_start': False,
    # Safety guard (see safety.py)
    'collision_margin': 0.01, 'safety_lookahead': 0.2, 'min_height': 0.03,
    'height_frames': ['tool0', 'camera_link', 'robotiq_85_left_finger_tip_link',
                      'robotiq_85_right_finger_tip_link'],
    # tool0 position limits in base_link: x_min, y_min, z_min, x_max, y_max, z_max
    'workspace': [0.15, -0.7, 0.0, 1.0, 0.7, 1.2],
}


class IbvsNode(Node):

    def __init__(self):
        super().__init__('ibvs_node')
        for name, default in PARAMS.items():
            self.declare_parameter(name, default)
        self.p = {name: self.get_parameter(name).value for name in PARAMS}
        self.add_on_set_parameters_callback(self.on_params)

        self.K = None
        self.features = None
        self.features_time = None
        self.depth = None
        self.q = None
        self.model = None
        self.safety = None
        self.active = False
        self.converged_since = None

        self.tf = Buffer()
        TransformListener(self.tf, self)

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, '/robot_description', self.on_description, latched)
        self.create_subscription(CameraInfo, self.p['camera_info_topic'], self.on_info,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, self.p['depth_topic'], self.on_depth,
                                 qos_profile_sensor_data)
        self.create_subscription(JointState, '/joint_states', self.on_joints, 10)
        self.create_subscription(ObjectFeatures, '/vs/features', self.on_features, 10)

        self.cmd_pub = self.create_publisher(Float64MultiArray, self.p['command_topic'], 10)
        self.twist_pub = self.create_publisher(TwistStamped, '~/camera_twist', 10)
        self.error_pub = self.create_publisher(Float64MultiArray, '~/error', 10)
        self.converged_pub = self.create_publisher(Bool, '~/converged', 10)

        self.switch = self.create_client(SwitchController, '/controller_manager/switch_controller')
        self.create_service(Trigger, '~/start', self.srv_start)
        self.create_service(Trigger, '~/stop', self.srv_stop)
        self.create_service(Trigger, '~/capture_reference', self.srv_capture)

        self.create_timer(1.0 / self.p['rate'], self.step)

    # --- inputs -------------------------------------------------------------------------------

    def on_params(self, changes):
        for c in changes:
            if c.name in self.p:
                self.p[c.name] = c.value
        return SetParametersResult(successful=True)

    def on_description(self, msg):
        full = pin.buildModelFromXML(msg.data)
        keep = set(self.p['arm_joints'])
        lock = [full.getJointId(n) for n in full.names[1:] if n not in keep]
        self.model = pin.buildReducedModel(full, lock, pin.neutral(full))
        self.data = self.model.createData()
        self.ee_id = self.model.getFrameId(self.p['ee_frame'])
        p = self.p
        self.safety = SafetyMonitor(msg.data, p['arm_joints'], margin=p['collision_margin'],
                                    lookahead=p['safety_lookahead'], min_height=p['min_height'],
                                    height_frames=p['height_frames'],
                                    workspace_frame=p['ee_frame'], workspace=p['workspace'])
        self.get_logger().info(f'kinematic model ready, {self.model.nq} joints, '
                               f'{len(self.safety.geom.collisionPairs)} collision pairs checked')

    def on_info(self, msg):
        self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def on_depth(self, msg):
        if msg.encoding == '32FC1':
            self.depth = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)

    def on_joints(self, msg):
        pos = dict(zip(msg.name, msg.position))
        if all(n in pos for n in self.p['arm_joints']):
            self.q = np.array([pos[n] for n in self.p['arm_joints']])

    def on_features(self, msg):
        self.features = msg
        self.features_time = self.get_clock().now()

    # --- services -----------------------------------------------------------------------------

    def switch_controllers(self, activate, deactivate, on_done):
        req = SwitchController.Request(activate_controllers=[activate],
                                       deactivate_controllers=[deactivate],
                                       strictness=SwitchController.Request.BEST_EFFORT)
        future = self.switch.call_async(req)
        future.add_done_callback(lambda f: on_done(f.result() is not None and f.result().ok))

    def start(self):
        def done(ok):
            self.active = ok
            self.converged_since = None
            self.get_logger().info('servoing started' if ok else 'controller switch failed')
        self.switch_controllers(self.p['velocity_controller'], self.p['position_controller'], done)

    def srv_start(self, _, response):
        self.start()
        response.success, response.message = True, 'switching to velocity control'
        return response

    def srv_stop(self, _, response):
        self.active = False
        self.publish_joint_velocities(np.zeros(len(self.p['arm_joints'])))
        self.switch_controllers(self.p['position_controller'], self.p['velocity_controller'],
                                lambda ok: self.get_logger().info('servoing stopped'))
        response.success, response.message = True, 'stopped'
        return response

    def srv_capture(self, _, response):
        """Use the current view as the reference image (desired pose)."""
        f = self.features
        if f is None or not f.valid:
            response.success, response.message = False, 'no valid target in view'
            return response
        self.p.update(desired_u=f.u, desired_v=f.v, desired_area=f.area, desired_angle=f.angle)
        if self.depth is not None:
            r, c = int(f.v), int(f.u)
            patch = self.depth[max(r - 3, 0):r + 4, max(c - 3, 0):c + 4]
            patch = patch[np.isfinite(patch)]
            if patch.size:
                self.p['desired_depth'] = float(np.median(patch))
        response.success = True
        response.message = (f'reference u={f.u:.1f} v={f.v:.1f} area={f.area:.0f} '
                            f'angle={f.angle:.2f} depth={self.p["desired_depth"]:.3f}')
        self.get_logger().info(response.message)
        return response

    # --- control loop -------------------------------------------------------------------------

    def publish_joint_velocities(self, qd):
        self.cmd_pub.publish(Float64MultiArray(data=[float(x) for x in qd]))

    def halt(self, reason):
        self.publish_joint_velocities(np.zeros(len(self.p['arm_joints'])))
        self.get_logger().warn(f'holding: {reason}', throttle_duration_sec=2.0)

    def target_ok(self):
        f = self.features
        if f is None or not f.valid:
            return False
        age = (self.get_clock().now() - self.features_time).nanoseconds * 1e-9
        return age <= self.p['feature_timeout']

    def step(self):
        ready = self.model is not None and self.K is not None and self.q is not None
        if not self.active:
            if self.p['auto_start'] and ready and self.target_ok():
                self.p['auto_start'] = False
                self.start()
            return
        if not ready:
            return self.halt('waiting for model, camera info or joint states')
        if not self.target_ok():
            return self.halt('target lost or features stale')
        try:
            T_ec = self.tf.lookup_transform(self.p['ee_frame'], self.p['camera_frame'], Time())
            T_bc = self.tf.lookup_transform(self.p['base_frame'], self.p['camera_frame'], Time())
        except Exception as exc:
            return self.halt(f'tf: {exc}')

        p, f = self.p, self.features
        u_star = p['desired_u'] if p['desired_u'] >= 0 else self.K[2]
        v_star = p['desired_v'] if p['desired_v'] >= 0 else self.K[3]
        s = ibvs.normalised_features(f.u, f.v, f.area, f.angle, self.K,
                                     p['desired_area'], p['desired_depth'])
        s_star = ibvs.normalised_features(u_star, v_star, p['desired_area'], p['desired_angle'],
                                          self.K, p['desired_area'], p['desired_depth'])
        v_cam, e = ibvs.control_law(s, s_star, p['gain'], p['control_yaw'])

        R_bc, _ = transform_to_rp(T_bc)
        if p['align_optical_axis_down']:
            # Rotate the optical axis (camera z) towards -z of the base; only wx, wy result
            w_base = p['align_gain'] * np.cross(R_bc[:, 2], [0.0, 0.0, -1.0])
            v_cam[3:5] += (R_bc.T @ w_base)[:2]
        v_cam[:3] = clamp_norm(v_cam[:3], p['max_linear_speed'])
        v_cam[3:] = clamp_norm(v_cam[3:], p['max_angular_speed'])

        # Hand-eye: camera twist -> tool0 twist, then into base axes for the Jacobian
        R_ec, p_ec = transform_to_rp(T_ec)
        v_ee = adjoint(R_ec, p_ec) @ v_cam
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        R_be = self.data.oMf[self.ee_id].rotation
        v_base = np.concatenate([R_be @ v_ee[:3], R_be @ v_ee[3:]])
        J = pin.computeFrameJacobian(self.model, self.data, self.q, self.ee_id,
                                     pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        qd = damped_pinv_solve(J, v_base, p['dls_damping'])
        qd = limit_joint_velocities(qd, self.q, self.model.lowerPositionLimit,
                                    self.model.upperPositionLimit,
                                    p['joint_speed_scale'] * self.model.velocityLimit,
                                    p['joint_limit_margin'])
        reason = self.safety.check(self.q, qd)
        if reason:
            return self.halt(reason)
        self.publish_joint_velocities(qd)
        self.publish_diagnostics(f.header, v_cam, e)

    def publish_diagnostics(self, header, v_cam, e):
        tw = TwistStamped()
        tw.header.stamp, tw.header.frame_id = header.stamp, self.p['camera_frame']
        tw.twist.linear.x, tw.twist.linear.y, tw.twist.linear.z = map(float, v_cam[:3])
        tw.twist.angular.x, tw.twist.angular.y, tw.twist.angular.z = map(float, v_cam[3:])
        self.twist_pub.publish(tw)
        self.error_pub.publish(Float64MultiArray(data=[float(x) for x in e]))

        now = self.get_clock().now()
        if np.linalg.norm(e) < self.p['converged_threshold']:
            self.converged_since = self.converged_since or now
        else:
            self.converged_since = None
        held = self.converged_since is not None and \
            (now - self.converged_since).nanoseconds * 1e-9 >= self.p['converged_hold_time']
        self.converged_pub.publish(Bool(data=held))


def main():
    rclpy.init()
    node = IbvsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
