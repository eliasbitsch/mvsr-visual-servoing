"""Frame and twist helpers. Twists are ordered (v, w) like pinocchio."""
import numpy as np


def skew(p):
    return np.array([[0, -p[2], p[1]], [p[2], 0, -p[0]], [-p[1], p[0], 0]])


def quat_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def transform_to_rp(tf_msg):
    t, q = tf_msg.transform.translation, tf_msg.transform.rotation
    return quat_to_matrix(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])


def adjoint(R, p):
    """Ad_T for T = (R, p), the pose of frame b in frame a: maps a twist of b given in b
    coordinates to the same rigid motion given in a coordinates at a's origin."""
    ad = np.zeros((6, 6))
    ad[:3, :3] = R
    ad[:3, 3:] = skew(p) @ R
    ad[3:, 3:] = R
    return ad


def damped_pinv_solve(J, twist, damping):
    """q_dot = J^T (J J^T + d^2 I)^-1 twist; stays bounded near singularities."""
    return J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(J.shape[0]), twist)


def limit_joint_velocities(qd, q, lower, upper, vmax, margin):
    """Scale qd uniformly into vmax (keeps the direction), then block motion into limits."""
    qd = qd / max(1.0, float(np.max(np.abs(qd) / vmax)))
    qd[(q > upper - margin) & (qd > 0)] = 0.0
    qd[(q < lower + margin) & (qd < 0)] = 0.0
    return qd


def clamp_norm(vec, max_norm):
    n = np.linalg.norm(vec)
    return vec if n <= max_norm or n == 0 else vec * (max_norm / n)
