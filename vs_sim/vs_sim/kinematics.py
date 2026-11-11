"""Kinematics helpers that work directly on the URDF (robot_description).

The chain is read from the URDF that robot_state_publisher publishes, so it
always matches the simulated robot (including the camera mount).

Example
-------
    chain = Chain(urdf_xml, base="world", tip="camera_optical_frame")
    T = chain.fk(q)          # 4x4 pose of the tip in the base frame
    J = chain.jacobian(q)    # 6xN geometric Jacobian [v; w] in the base frame
"""

import xml.etree.ElementTree as ET

import numpy as np


# --------------------------------------------------------------------------- #
# Basic SE(3)/SO(3) helpers
# --------------------------------------------------------------------------- #

def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rpy_to_matrix(roll, pitch, yaw):
    """URDF convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def matrix_to_rpy(R):
    pitch = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) > 1e-9:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    return np.array([roll, pitch, yaw])


def axis_angle_to_matrix(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def make_tf(R=None, t=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if t is not None:
        T[:3, 3] = t
    return T


def inv_tf(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def rotation_log(R):
    """Rotation vector (axis * angle) of R, robust near 0 and pi."""
    cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_a)
    if angle < 1e-9:
        return np.zeros(3)
    if np.pi - angle < 1e-6:
        # axis from the diagonal of (R + I) / 2
        M = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(M), 0.0, None))
        i = int(np.argmax(axis))
        axis[(i + 1) % 3] = np.copysign(axis[(i + 1) % 3], M[i, (i + 1) % 3])
        axis[(i + 2) % 3] = np.copysign(axis[(i + 2) % 3], M[i, (i + 2) % 3])
        return axis / np.linalg.norm(axis) * angle
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2.0 * np.sin(angle)) * angle


def quat_to_matrix(x, y, z, w):
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R):
    """Returns (x, y, z, w)."""
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def pose_msg_to_tf(pose):
    """geometry_msgs/Pose -> 4x4."""
    p, o = pose.position, pose.orientation
    return make_tf(quat_to_matrix(o.x, o.y, o.z, o.w), [p.x, p.y, p.z])


def transform_msg_to_tf(tr):
    """geometry_msgs/Transform -> 4x4."""
    t, o = tr.translation, tr.rotation
    return make_tf(quat_to_matrix(o.x, o.y, o.z, o.w), [t.x, t.y, t.z])


def tf_to_pose_msg(T, pose):
    """Fill a geometry_msgs/Pose from a 4x4 and return it."""
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in T[:3, 3])
    q = matrix_to_quat(T[:3, :3])
    pose.orientation.x, pose.orientation.y = float(q[0]), float(q[1])
    pose.orientation.z, pose.orientation.w = float(q[2]), float(q[3])
    return pose


def pose_error(T_current, T_desired):
    """Error of T_current w.r.t. T_desired, expressed in the desired frame.

    Returns (translation_vector, rotation_vector).
    """
    T_err = inv_tf(T_desired) @ T_current
    return T_err[:3, 3].copy(), rotation_log(T_err[:3, :3])


# --------------------------------------------------------------------------- #
# URDF chain
# --------------------------------------------------------------------------- #

class _Segment:
    def __init__(self, name, jtype, parent, child, origin, axis, lower, upper):
        self.name = name
        self.type = jtype
        self.parent = parent
        self.child = child
        self.origin = origin
        self.axis = axis
        self.lower = lower
        self.upper = upper

    @property
    def movable(self):
        return self.type in ("revolute", "continuous", "prismatic")


def _parse_origin(el):
    if el is None:
        return np.eye(4)
    xyz = [float(v) for v in el.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in el.get("rpy", "0 0 0").split()]
    return make_tf(rpy_to_matrix(*rpy), xyz)


class Chain:
    """Serial kinematic chain between two links of a URDF."""

    def __init__(self, urdf_xml, base="world", tip="tool0"):
        root = ET.fromstring(urdf_xml)
        by_child = {}
        for j in root.findall("joint"):
            axis_el = j.find("axis")
            axis = np.array([float(v) for v in axis_el.get("xyz").split()]) \
                if axis_el is not None else np.array([1.0, 0.0, 0.0])
            lim = j.find("limit")
            lower = float(lim.get("lower", -np.inf)) if lim is not None else -np.inf
            upper = float(lim.get("upper", np.inf)) if lim is not None else np.inf
            if j.get("type") == "continuous":
                lower, upper = -np.inf, np.inf
            seg = _Segment(j.get("name"), j.get("type"), j.find("parent").get("link"),
                           j.find("child").get("link"), _parse_origin(j.find("origin")),
                           axis, lower, upper)
            by_child[seg.child] = seg

        segments = []
        link = tip
        while link != base:
            if link not in by_child:
                raise ValueError(f"No chain from '{base}' to '{tip}' (stopped at '{link}')")
            seg = by_child[link]
            segments.append(seg)
            link = seg.parent
        segments.reverse()

        self.base = base
        self.tip = tip
        self.segments = segments
        self.joint_names = [s.name for s in segments if s.movable]
        self.lower = np.array([s.lower for s in segments if s.movable])
        self.upper = np.array([s.upper for s in segments if s.movable])

    @property
    def n(self):
        return len(self.joint_names)

    def q_from_joint_state(self, msg):
        """Joint vector in chain order from a sensor_msgs/JointState."""
        lookup = dict(zip(msg.name, msg.position))
        return np.array([lookup[n] for n in self.joint_names])

    def _frames(self, q):
        q = np.asarray(q, dtype=float)
        T = np.eye(4)
        axes = []
        i = 0
        for s in self.segments:
            T = T @ s.origin
            if s.movable:
                axis_world = T[:3, :3] @ s.axis
                axes.append((s.type, axis_world, T[:3, 3].copy()))
                if s.type == "prismatic":
                    T = T @ make_tf(t=s.axis * q[i])
                else:
                    T = T @ make_tf(axis_angle_to_matrix(s.axis, q[i]))
                i += 1
        return T, axes

    def fk(self, q):
        """Pose of the tip link in the base frame (4x4)."""
        return self._frames(q)[0]

    def jacobian(self, q):
        """Geometric Jacobian (6xN) of the tip origin in the base frame.

        Rows 0-2: linear velocity, rows 3-5: angular velocity.
        """
        T, axes = self._frames(q)
        p_tip = T[:3, 3]
        J = np.zeros((6, self.n))
        for i, (jtype, z, p) in enumerate(axes):
            if jtype == "prismatic":
                J[:3, i] = z
            else:
                J[:3, i] = np.cross(z, p_tip - p)
                J[3:, i] = z
        return J

    def ik(self, T_goal, q0, max_iter=400, tol_pos=1e-5, tol_rot=1e-4, damping=0.02, step=0.7):
        """Damped-least-squares IK. Returns (q, success)."""
        q = np.array(q0, dtype=float)
        for _ in range(max_iter):
            T = self.fk(q)
            e_p = T_goal[:3, 3] - T[:3, 3]
            e_r = rotation_log(T_goal[:3, :3] @ T[:3, :3].T)
            if np.linalg.norm(e_p) < tol_pos and np.linalg.norm(e_r) < tol_rot:
                return q, True
            e = np.concatenate([e_p, e_r])
            J = self.jacobian(q)
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
            q = np.clip(q + step * dq, self.lower, self.upper)
        T = self.fk(q)
        ok = (np.linalg.norm(T_goal[:3, 3] - T[:3, 3]) < 10 * tol_pos and
              np.linalg.norm(rotation_log(T_goal[:3, :3] @ T[:3, :3].T)) < 10 * tol_rot)
        return q, ok
