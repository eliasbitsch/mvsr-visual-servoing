"""Target and start-configuration definitions (config/targets.yaml)."""

import os

import numpy as np
import yaml

from vs_sim.kinematics import make_tf, rot_x, rot_z, rpy_to_matrix

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:  # allows offline tests without ROS
    get_package_share_directory = None


def default_config_path():
    if get_package_share_directory is not None:
        try:
            return os.path.join(get_package_share_directory("vs_sim"), "config", "targets.yaml")
        except Exception:
            pass
    return os.path.join(os.path.dirname(__file__), "..", "config", "targets.yaml")


def load_config(path=None):
    with open(path or default_config_path(), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_pose_from_config(cfg, target):
    """Nominal model pose (4x4, world) as written in targets.yaml."""
    x, y, z, yaw = cfg["targets"][target]["pose"]
    return make_tf(rot_z(yaw), [x, y, z])


def target_frame(cfg, target, T_world_model):
    """Target frame = model frame shifted to the top surface of the object."""
    top = float(cfg["targets"][target].get("top", 0.0))
    return T_world_model @ make_tf(t=[0.0, 0.0, top])


def desired_camera_in_target(cfg):
    """Desired pose of camera_optical_frame in the target frame.

    The camera looks straight down onto the target (optical z = -z_target),
    at the configured distance above the top surface.
    """
    d = float(cfg["reference"]["distance"])
    return make_tf(rot_x(np.pi), [0.0, 0.0, d])


def start_offsets(cfg):
    return cfg["start_configurations"]


def offset_tf(offset):
    """[dx, dy, dz, droll, dpitch, dyaw] -> (T_trans in target frame, R in camera frame)."""
    dx, dy, dz, r, p, y = offset
    T_trans = make_tf(t=[dx, dy, dz])
    T_rot = make_tf(rpy_to_matrix(np.radians(r), np.radians(p), np.radians(y)))
    return T_trans, T_rot


def camera_goal_in_world(cfg, T_world_target, offset=None):
    """World pose for camera_optical_frame: reference pose plus an optional offset.

    The translation part of the offset is applied in the target frame
    (z = up), the rotation part about the axes of the desired camera frame.
    """
    T_t_c = desired_camera_in_target(cfg)
    if offset is None:
        return T_world_target @ T_t_c
    T_trans, T_rot = offset_tf(offset)
    return T_world_target @ T_trans @ T_t_c @ T_rot
