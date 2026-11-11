"""IBVS with decoupled region features for a planar target roughly parallel to the image plane.

Features (Tahri & Chaumette 2005, image moments for visual servoing):
  a_n = Z* sqrt(a* / a)      normalised area, behaves like the depth
  x_n = a_n x,  y_n = a_n y  normalised centroid (x, y in normalised image coordinates)
  alpha                      principal-axis angle
With these, the interaction matrix for (vx, vy, vz, wz) is close to constant, so the motion is
nearly straight in Cartesian space. wx and wy are not servoed from the image here, the node keeps
the optical axis vertical with forward kinematics instead.
"""
import numpy as np


def normalised_features(u, v, area, angle, K, desired_area, desired_depth):
    fx, fy, cx, cy = K
    a_n = desired_depth * np.sqrt(desired_area / max(area, 1.0))
    return np.array([a_n * (u - cx) / fx, a_n * (v - cy) / fy, a_n, angle])


def interaction_matrix(s):
    """Rows (x_n, y_n, a_n, alpha), columns (vx, vy, vz, wz)."""
    x_n, y_n = s[0], s[1]
    return np.array([
        [-1.0, 0.0, 0.0, y_n],
        [0.0, -1.0, 0.0, -x_n],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
    ])


def wrap_angle(a):
    # Principal axis is only defined modulo pi
    return (a + np.pi / 2) % np.pi - np.pi / 2


def control_law(s, s_star, gain, control_yaw):
    """v = -gain L^+ e. Returns the camera twist (vx, vy, vz, wx, wy, wz) and the error."""
    e = s - s_star
    e[3] = wrap_angle(e[3])
    L = interaction_matrix(s)
    if not control_yaw:
        e, L = e[:3], L[:3, :3]
    v = -gain * np.linalg.pinv(L) @ e
    twist = np.zeros(6)
    twist[:3] = v[:3]
    if control_yaw:
        twist[5] = v[3]
    return twist, e
