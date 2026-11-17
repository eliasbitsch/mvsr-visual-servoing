"""Offline consistency tests (no ROS/Gazebo needed): pytest test/"""

import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import yaml

from vs_sim.kinematics import (Chain, inv_tf, make_tf, matrix_to_quat, quat_to_matrix,
                               rotation_log, rpy_to_matrix)

PKG = os.path.join(os.path.dirname(__file__), "..")
CFG = yaml.safe_load(open(os.path.join(PKG, "config", "targets.yaml")))


def test_world_matches_targets_yaml():
    world = ET.parse(os.path.join(PKG, "worlds", "vs_world.sdf")).getroot().find("world")
    poses = {inc.find("name").text: [float(v) for v in inc.find("pose").text.split()]
             for inc in world.findall("include")}
    for name, t in CFG["targets"].items():
        x, y, z, yaw = t["pose"]
        p = poses[t["model"]]
        assert np.allclose(p, [x, y, z, 0, 0, yaw]), name
        assert os.path.isfile(os.path.join(PKG, "models", t["model"], "model.sdf")), name


def test_model_files_are_valid_xml():
    for root, _, files in os.walk(os.path.join(PKG, "models")):
        for f in files:
            if f.endswith((".sdf", ".config")):
                ET.parse(os.path.join(root, f))


def test_aruco_texture_detectable():
    img = cv2.imread(os.path.join(PKG, "models", "aruco_target", "materials", "textures",
                                  "aruco_4x4_50_id0.png"))
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "ArucoDetector"):
        _, ids, _ = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters()).detectMarkers(img)
    else:  # OpenCV < 4.7 (Ubuntu 24.04 apt)
        _, ids, _ = cv2.aruco.detectMarkers(img, d, parameters=cv2.aruco.DetectorParameters_create())
    assert ids is not None and ids.flatten().tolist() == [0]


def test_so3_helpers():
    rng = np.random.default_rng(1)
    for _ in range(50):
        R = rpy_to_matrix(*rng.uniform(-3, 3, 3))
        assert np.allclose(quat_to_matrix(*matrix_to_quat(R)), R, atol=1e-9)
        w = rotation_log(R)
        K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        a = np.linalg.norm(w)
        R2 = np.eye(3) + np.sin(a) / a * K + (1 - np.cos(a)) / a ** 2 * K @ K
        assert np.allclose(R, R2, atol=1e-6)
    T = make_tf(rpy_to_matrix(0.3, -0.2, 1.0), [1, 2, 3])
    assert np.allclose(inv_tf(T) @ T, np.eye(4))


URDF = """<robot name="t">
  <link name="world"/><link name="a"/><link name="b"/><link name="tip"/>
  <joint name="j1" type="revolute"><parent link="world"/><child link="a"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="j2" type="revolute"><parent link="a"/><child link="b"/>
    <origin xyz="0.4 0 0" rpy="1.5708 0 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>
  <joint name="f" type="fixed"><parent link="b"/><child link="tip"/>
    <origin xyz="0.2 0.05 0" rpy="0 0.4 0"/></joint>
</robot>"""


def test_jacobian_matches_finite_differences():
    ch = Chain(URDF, "world", "tip")
    q = np.array([0.4, -0.7])
    J = ch.jacobian(q)
    T0, eps = ch.fk(q), 1e-6
    for i in range(2):
        dq = np.zeros(2)
        dq[i] = eps
        T = ch.fk(q + dq)
        assert np.allclose((T[:3, 3] - T0[:3, 3]) / eps, J[:3, i], atol=1e-5)
        assert np.allclose(rotation_log(T[:3, :3] @ T0[:3, :3].T) / eps, J[3:, i], atol=1e-5)
