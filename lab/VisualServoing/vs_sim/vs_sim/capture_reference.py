"""Captures the reference image (desired view) for a target.

    ros2 run vs_sim capture_reference --target aruco
    ros2 run vs_sim capture_reference --all

Moves the camera into the reference pose and stores in ~/vs_data/reference/:
    <target>.png    camera image in the reference pose
    <target>.yaml   joint configuration, camera intrinsics, desired camera pose
                    in the target frame, and (for the ArUco target) the
                    detected marker corners = desired image features
"""

import argparse
import os
import sys

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image

from vs_sim import targets as tg


def detect_aruco(img):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2.aruco, "ArucoDetector"):
        corners, ids, _ = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters()).detectMarkers(gray)
    else:  # OpenCV < 4.7
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=cv2.aruco.DetectorParameters_create())
    return corners, ids


def capture(mover, target, out_dir):
    q, _ = mover.go_to(target, None, mode="velocity", settle=1.0)
    bridge = CvBridge()
    box = {}
    subs = [mover.create_subscription(Image, "/camera/image_raw", lambda m: box.__setitem__("img", m), 1),
            mover.create_subscription(CameraInfo, "/camera/camera_info", lambda m: box.__setitem__("info", m), 1)]
    mover.sleep(0.5)
    box.clear()
    mover._spin_until(lambda: "img" in box and "info" in box, 10.0, "/camera/image_raw and /camera/camera_info")
    for s in subs:
        mover.destroy_subscription(s)

    img = bridge.imgmsg_to_cv2(box["img"], desired_encoding="bgr8")
    info = box["info"]
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, f"{target}.png"), img)

    T_tc = tg.desired_camera_in_target(mover.cfg)
    data = {
        "target": target,
        "description": mover.cfg["targets"][target].get("description", ""),
        "joint_names": list(mover.chain.joint_names),
        "joint_positions": [float(v) for v in q],
        "camera": {"width": int(info.width), "height": int(info.height),
                   "K": [float(v) for v in info.k], "D": [float(v) for v in info.d],
                   "frame_id": info.header.frame_id},
        "desired_camera_in_target": {"matrix": np.round(T_tc, 6).tolist()},
    }
    if target == "aruco":
        corners, ids = detect_aruco(img)
        if ids is not None and len(ids) > 0:
            data["aruco"] = {"id": int(ids[0][0]),
                             "corners_px": np.round(corners[0].reshape(4, 2), 2).tolist(),
                             "marker_size": float(mover.cfg["targets"]["aruco"]["marker_size"])}
        else:
            mover.get_logger().warn(
                "ArUco marker NOT detected in the reference image. If the marker looks mirrored, "
                "regenerate the textures with 'tools/generate_textures.py --mirror'.")
    with open(os.path.join(out_dir, f"{target}.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    mover.get_logger().info(f"Saved {out_dir}/{target}.png and {target}.yaml")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="aruco")
    ap.add_argument("--all", action="store_true", help="capture all targets")
    ap.add_argument("--out", default=os.path.expanduser("~/vs_data/reference"))
    args = ap.parse_args(rclpy.utilities.remove_ros_args(argv if argv is not None else sys.argv)[1:])

    from vs_sim.motion import RobotMover
    rclpy.init()
    mover = RobotMover("capture_reference")
    try:
        names = list(mover.cfg["targets"]) if args.all else [args.target]
        for name in names:
            capture(mover, name, args.out)
    except Exception as e:  # noqa: BLE001
        mover.get_logger().error(str(e))
        sys.exit(1)
    finally:
        mover.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
