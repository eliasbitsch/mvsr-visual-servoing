"""Moves the robot to a start configuration or the reference pose.

    ros2 run vs_sim go_to_start --list
    ros2 run vs_sim go_to_start --target aruco --start 3
    ros2 run vs_sim go_to_start --target red_can --start S6_rotation_90
    ros2 run vs_sim go_to_start --target aruco --reference
    ros2 run vs_sim go_to_start --target aruco --start 1 --mode trajectory

Afterwards the forward_velocity_controller is active (default) and the robot
stands still, ready for your visual servoing controller.
"""

import argparse
import sys

import rclpy

from vs_sim import targets as tg


def print_list(cfg):
    print("Targets:")
    for name, t in cfg["targets"].items():
        print(f"  {name:12s} {t.get('description', '')}")
    print("\nStart configurations  [dx dy dz (m, target frame) | droll dpitch dyaw (deg, camera frame)]:")
    for i, s in enumerate(tg.start_offsets(cfg), 1):
        print(f"  {i:2d}  {s['name']:18s} {s['offset']}")
    print(f"\nReference: camera {cfg['reference']['distance']:.2f} m above the target, looking straight down")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="aruco")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--start", help="number (1-based) or name of the start configuration")
    g.add_argument("--reference", action="store_true", help="go to the reference (goal) pose")
    ap.add_argument("--mode", choices=["velocity", "trajectory"], default="velocity",
                    help="controller that is active afterwards")
    ap.add_argument("--list", action="store_true", help="list targets and start configurations")
    ap.add_argument("--targets-file", help="alternative targets.yaml")
    args = ap.parse_args(rclpy.utilities.remove_ros_args(argv if argv is not None else sys.argv)[1:])

    if args.list:
        print_list(tg.load_config(args.targets_file))
        return
    if not args.reference and args.start is None:
        ap.error("use --start N, --reference or --list")

    from vs_sim.motion import RobotMover
    rclpy.init()
    mover = RobotMover("go_to_start", args.targets_file)
    try:
        _, name = mover.go_to(args.target, None if args.reference else args.start, mode=args.mode)
        mover.get_logger().info(f"Reached '{args.target}' / '{name}', active controller: "
                                f"{'forward_velocity_controller' if args.mode == 'velocity' else 'scaled_joint_trajectory_controller'}")
    except Exception as e:  # noqa: BLE001
        mover.get_logger().error(str(e))
        sys.exit(1)
    finally:
        mover.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
