"""Offline check of all targets x start configurations (no simulation needed).

For every target and start configuration it checks that
  * an IK solution exists (seeded from the reference pose of that target),
  * the robot is not close to a singularity,
  * the target footprint is completely inside the camera image.

Usage:
    ros2 run vs_sim check_start_configs [--urdf FILE] [--targets FILE]
"""

import argparse
import subprocess
import sys

import numpy as np

from vs_sim import targets as tg
from vs_sim.kinematics import Chain, inv_tf

# Reference joint seed (camera above the ArUco target), see urdf/ur_vs.urdf.xacro
Q_HOME = np.array([-0.301, -1.629, 2.069, -2.011, -1.571, -0.301])


def build_urdf(ur_type="ur5e"):
    from ament_index_python.packages import get_package_share_directory
    xacro_file = get_package_share_directory("vs_sim") + "/urdf/ur_vs.urdf.xacro"
    return subprocess.check_output(["xacro", xacro_file, f"ur_type:={ur_type}", "name:=ur"], text=True)


def camera_intrinsics(hfov=1.0472, width=640, height=480):
    f = (width / 2.0) / np.tan(hfov / 2.0)
    return f, width / 2.0, height / 2.0, width, height


def check(urdf, cfg, verbose=True):
    cam = Chain(urdf, "world", "camera_optical_frame")
    f, cx, cy, w, h = camera_intrinsics()
    problems = 0
    for name, t in cfg["targets"].items():
        T_wt = tg.target_frame(cfg, name, tg.model_pose_from_config(cfg, name))
        q_ref, ok_ref = cam.ik(tg.camera_goal_in_world(cfg, T_wt), Q_HOME, max_iter=3000)
        hx, hy = (t.get("footprint", [0.05, 0.05])[0] / 2.0, t.get("footprint", [0.05, 0.05])[1] / 2.0)
        if verbose:
            print(f"{name}: reference IK {'ok' if ok_ref else 'FAILED'}")
        for s in [{"name": "reference", "offset": None}] + tg.start_offsets(cfg):
            q, ok = cam.ik(tg.camera_goal_in_world(cfg, T_wt, s["offset"]), q_ref, max_iter=3000)
            sv = np.linalg.svd(cam.jacobian(q), compute_uv=False).min()
            T_ct = inv_tf(cam.fk(q)) @ T_wt
            uv = []
            for sx in (-1, 1):
                for sy in (-1, 1):
                    p = T_ct @ np.array([sx * hx, sy * hy, 0.0, 1.0])
                    uv.append((f * p[0] / p[2] + cx, f * p[1] / p[2] + cy))
            uv = np.array(uv)
            visible = uv[:, 0].min() > 0 and uv[:, 0].max() < w and uv[:, 1].min() > 0 and uv[:, 1].max() < h
            good = ok and visible and sv > 0.05
            problems += 0 if good else 1
            if verbose:
                print(f"   {s['name']:18s} ik={'ok' if ok else 'FAIL'} visible={visible} "
                      f"sigma_min={sv:.3f} u=[{uv[:, 0].min():.0f},{uv[:, 0].max():.0f}] "
                      f"v=[{uv[:, 1].min():.0f},{uv[:, 1].max():.0f}]{'' if good else '   <-- PROBLEM'}")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", help="URDF file (default: process urdf/ur_vs.urdf.xacro)")
    ap.add_argument("--targets", help="targets.yaml (default: installed config)")
    args = ap.parse_args(argv)
    urdf = open(args.urdf).read() if args.urdf else build_urdf()
    problems = check(urdf, tg.load_config(args.targets))
    print("OK" if problems == 0 else f"{problems} problem(s)")
    sys.exit(0 if problems == 0 else 1)


if __name__ == "__main__":
    main()
