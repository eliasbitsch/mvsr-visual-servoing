"""Evaluation runner (assignment section 2.5): every target x start configuration, automatically.

Per target: set the prompt, move to the reference pose, capture the reference features. Per start
configuration: move there, wait until the target is seen, start servoing, record until converged or
timeout, stop. The ground truth is only recorded for the metrics, the controller never sees it.

  ros2 run vs_evaluation run_eval --targets red_can blue_box --starts 1 2 3 --label dino_csrt

Output in /root/vs_data/eval/<time>_<label>/: summary.csv (one row per run), runs/<target>_S<n>.csv
(time series), config.json.
"""
import argparse
import csv
import json
import math
import os
import time

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Bool, Float64MultiArray
from std_srvs.srv import Trigger
from vs_interfaces.msg import ObjectFeatures
from vs_sim.motion import RobotMover

# Prompt and rotational symmetry seen from above: 'round' has no yaw, 'pi' is identical after 180 deg
TARGETS = {
    'red_can': ('red can', 'round'),
    'blue_box': ('blue box', 'pi'),
    'green_cup': ('green cup', 'round'),
    'yellow_ball': ('yellow ball', 'round'),
}

SERIES_FIELDS = (['t', 'converged', 'feat_valid', 'feat_source', 'u', 'v', 'area', 'angle',
                  'e_x', 'e_y', 'e_area', 'e_angle',
                  'v_x', 'v_y', 'v_z', 'w_x', 'w_y', 'w_z',
                  'gt_ex', 'gt_ey', 'gt_ez', 'gt_rx', 'gt_ry', 'gt_rz', 'gt_et', 'gt_er',
                  'cam_x', 'cam_y', 'cam_z', 'cam_qx', 'cam_qy', 'cam_qz', 'cam_qw'])

SUMMARY_FIELDS = ['target', 'start', 'success', 'converged', 'conv_time_s', 'outcome',
                  'final_pos_err_m', 'final_rot_err_deg', 'final_rot_err_raw_deg',
                  'final_feature_err', 'detector_calls']


def symmetric_rot_err(rx, ry, rz, symmetry):
    """Rotation error that ignores what the target's symmetry makes invisible (small-tilt approx.)."""
    if symmetry == 'round':
        return math.hypot(rx, ry)
    if symmetry == 'pi':
        rz = (rz + math.pi / 2) % math.pi - math.pi / 2
    return math.sqrt(rx * rx + ry * ry + rz * rz)


class Recorder(Node):
    """Keeps the latest value of every topic and records one row per controller error message."""

    def __init__(self):
        super().__init__('vs_eval_recorder')
        self.feat = self.gt = self.cam = self.twist = None
        self.converged = False
        self.clock_seen = 0.0
        self.rows = None
        self.t0 = 0.0
        self.detections = 0
        self.create_subscription(ObjectFeatures, '/vs/features', self.on_feat, 10)
        self.create_subscription(Float64MultiArray, '/ground_truth/pose_error',
                                 lambda m: setattr(self, 'gt', list(m.data)), 10)
        self.create_subscription(PoseStamped, '/ground_truth/camera_pose',
                                 lambda m: setattr(self, 'cam', m.pose), 10)
        self.create_subscription(TwistStamped, '/ibvs_node/camera_twist',
                                 lambda m: setattr(self, 'twist', m.twist), 10)
        self.create_subscription(Bool, '/ibvs_node/converged',
                                 lambda m: setattr(self, 'converged', m.data), 10)
        self.create_subscription(Float64MultiArray, '/ibvs_node/error', self.on_error, 10)
        self.create_subscription(Clock, '/clock',
                                 lambda _: setattr(self, 'clock_seen', time.monotonic()),
                                 qos_profile_sensor_data)
        self.cli = {n: self.create_client(Trigger, f'/ibvs_node/{n}')
                    for n in ('start', 'stop', 'capture_reference')}
        self.param_cli = self.create_client(SetParameters, '/perception_node/set_parameters')

    def on_feat(self, m):
        self.feat = m
        if m.source == 'detector':
            self.detections += 1

    def on_error(self, m):
        if self.rows is None:
            return
        e = list(m.data) + [math.nan] * (4 - len(m.data))
        f, g, c, w = self.feat, self.gt or [math.nan] * 8, self.cam, self.twist
        row = [time.monotonic() - self.t0, int(self.converged),
               int(bool(f and f.valid)), f.source if f else '',
               f.u if f else math.nan, f.v if f else math.nan,
               f.area if f else math.nan, f.angle if f else math.nan, *e[:4]]
        row += ([w.linear.x, w.linear.y, w.linear.z, w.angular.x, w.angular.y, w.angular.z]
                if w else [math.nan] * 6)
        row += g[:8]
        row += ([c.position.x, c.position.y, c.position.z, c.orientation.x, c.orientation.y,
                 c.orientation.z, c.orientation.w] if c else [math.nan] * 7)
        self.rows.append(row)

    def check_sim_clock(self):
        # A stalled /clock freezes the trajectory controller: goals are accepted, never executed
        self.spin_for(1.0)
        if time.monotonic() - self.clock_seen > 1.0:
            raise RuntimeError('no /clock from the simulation: restart the sim container')

    def spin_for(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def call(self, name, timeout=10.0):
        cli = self.cli[name]
        if not cli.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f'/ibvs_node/{name} not available')
        fut = cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        return fut.result()

    def set_prompt(self, prompt):
        if not self.param_cli.wait_for_service(timeout_sec=10.0):
            raise RuntimeError('/perception_node/set_parameters not available')
        value = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=prompt)
        fut = self.param_cli.call_async(
            SetParameters.Request(parameters=[Parameter(name='prompt', value=value)]))
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)

    def wait_target(self, prompt, timeout, hold=1.0):
        """Wait until the target with this prompt is seen continuously for `hold` seconds.

        Detector or tracker, both count: with detect_on_loss the detector only runs while no
        track exists, so after a move the tracker may be the only source.
        """
        self.feat, since, end = None, None, time.monotonic() + timeout
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            f = self.feat
            if f and f.valid and f.prompt == prompt:
                since = since or time.monotonic()
                if time.monotonic() - since >= hold:
                    return True
            elif f is not None:
                since = None
        return False


def run_one(rec, symmetry, timeout):
    """Servo from the current pose. Returns the summary values and the recorded rows."""
    rec.spin_for(0.3)  # drop converged messages still queued from the previous run
    rec.rows, rec.t0, rec.converged, rec.detections = [], time.monotonic(), False, 0
    rec.call('start')
    conv_time = None
    while time.monotonic() - rec.t0 < timeout:
        rclpy.spin_once(rec, timeout_sec=0.05)
        if rec.converged:
            conv_time = time.monotonic() - rec.t0
            break
        if rec.feat is not None and not rec.feat.valid and time.monotonic() - rec.t0 > 2.0:
            break
    rec.spin_for(0.5)  # a few rows in the final pose
    rec.call('stop')
    rows, rec.rows = rec.rows, None
    g = rec.gt or [math.nan] * 8
    pos_err = g[6]
    rot_err = symmetric_rot_err(g[3], g[4], g[5], symmetry)
    last = rows[-1] if rows else None
    feat_err = (math.sqrt(sum(x * x for x in last[8:12] if not math.isnan(x)))
                if last else math.nan)
    if conv_time is not None:
        outcome = 'converged'
    elif rec.feat is not None and not rec.feat.valid:
        outcome = 'target lost'
    else:
        outcome = 'timeout'
    return {'converged': int(conv_time is not None),
            'conv_time_s': round(conv_time, 2) if conv_time is not None else '',
            'outcome': outcome,
            'final_pos_err_m': round(pos_err, 5),
            'final_rot_err_deg': round(math.degrees(rot_err), 3),
            'final_rot_err_raw_deg': round(math.degrees(g[7]), 3),
            'final_feature_err': round(feat_err, 5),
            'detector_calls': rec.detections}, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--targets', nargs='+', default=list(TARGETS), choices=list(TARGETS))
    ap.add_argument('--starts', nargs='+', type=int, default=list(range(1, 9)))
    ap.add_argument('--timeout', type=float, default=60.0, help='seconds per run')
    ap.add_argument('--pos-tol', type=float, default=0.005, help='success tolerance [m]')
    ap.add_argument('--rot-tol', type=float, default=2.0, help='success tolerance [deg]')
    ap.add_argument('--label', default='run', help='name part of the output folder')
    ap.add_argument('--out', default='/root/vs_data/eval')
    args = ap.parse_args()

    out = os.path.join(args.out, time.strftime('%Y%m%d_%H%M%S') + '_' + args.label)
    os.makedirs(os.path.join(out, 'runs'))
    with open(os.path.join(out, 'config.json'), 'w') as fh:
        json.dump(vars(args), fh, indent=2)

    rclpy.init()
    mover, rec = RobotMover(), Recorder()
    mover.ready()
    with open(os.path.join(out, 'summary.csv'), 'w', newline='') as fh:
        summary = csv.DictWriter(fh, SUMMARY_FIELDS)
        summary.writeheader()
        for target in args.targets:
            prompt, symmetry = TARGETS[target]
            rec.call('stop')
            rec.set_prompt(prompt)
            rec.check_sim_clock()
            mover.go_to(target, start=None)
            if not rec.wait_target(prompt, 30.0) or not rec.call('capture_reference').success:
                print(f'{target}: no detection at the reference pose, skipped', flush=True)
                continue
            for start in args.starts:
                rec.check_sim_clock()
                mover.go_to(target, start=start)
                row = {'target': target, 'start': start}
                if not rec.wait_target(prompt, 20.0):
                    row.update(success=0, converged=0, outcome='not detected')
                    rows = []
                else:
                    result, rows = run_one(rec, symmetry, args.timeout)
                    row.update(result)
                    row['success'] = int(result['converged'] and
                                         result['final_pos_err_m'] < args.pos_tol and
                                         result['final_rot_err_deg'] < args.rot_tol)
                summary.writerow(row)
                fh.flush()
                with open(os.path.join(out, 'runs', f'{target}_S{start}.csv'), 'w',
                          newline='') as rf:
                    w = csv.writer(rf)
                    w.writerow(SERIES_FIELDS)
                    w.writerows(rows)
                print(f'{target} S{start}: {row.get("outcome")} success={row["success"]} '
                      f'pos={row.get("final_pos_err_m", "")} rot={row.get("final_rot_err_deg", "")} '
                      f't={row.get("conv_time_s", "")}', flush=True)
    rec.call('stop')
    print(f'results in {out}', flush=True)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
