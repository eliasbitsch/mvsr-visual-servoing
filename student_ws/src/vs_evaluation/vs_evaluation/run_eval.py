"""Evaluation runner (assignment sections 2.4 and 2.5): every target x start configuration.

Per target: set the prompt, move to the reference pose, capture the reference features (always
without disturbance). Then optionally apply a disturbance (occluder plate, light level). Per start
configuration: move there, save camera image + ground truth for the offline prompt comparison,
wait until the target is seen, servo until converged or timeout, stop. The ground truth is only
recorded for the metrics and the image set; the controller never sees it.

  ros2 run vs_evaluation run_eval --label dino_sam2
  ros2 run vs_evaluation run_eval --occlusion 0.4 --label occl40
  ros2 run vs_evaluation run_eval --light 0.3 --label dark
  ros2 run vs_evaluation run_eval --detector-period 5 --detector-delay 1.0 --label slow_fm
  (fm launched with tracker:=none) ros2 run vs_evaluation run_eval --feature-timeout 1.5 --label no_tracker
  ros2 run vs_evaluation run_eval --lose-target-after 3 --starts 1 --label lost

Output in /root/vs_data/eval/<time>_<label>/: summary.csv (one row per run), runs/<target>_S<n>.csv
(time series at 20 Hz), images/ (camera frames with ground truth), config.json.
"""
import argparse
import csv
import json
import math
import os
import subprocess
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image
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

# The two lights of vs_world.sdf. light_config replaces the whole light, so every field is restated;
# without intensity Gazebo uses 0, which switches the light off.
LIGHTS = {
    'sun': 'type: DIRECTIONAL, diffuse: {r: 0.85, g: 0.85, b: 0.85, a: 1}, '
           'specular: {r: 0.2, g: 0.2, b: 0.2, a: 1}, direction: {x: -0.3, y: 0.2, z: -0.9}, '
           'cast_shadows: true',
    'lab_light': 'type: POINT, diffuse: {r: 0.4, g: 0.4, b: 0.4, a: 1}, '
                 'specular: {r: 0.1, g: 0.1, b: 0.1, a: 1}, range: 6, attenuation_constant: 0.6, '
                 'attenuation_linear: 0.05, attenuation_quadratic: 0.01',
}
OCCLUDER = 'vs_occluder'
OCCLUDER_HEIGHT = 0.08   # above the target top, between camera (0.30 above) and target
SAMPLE_RATE = 20.0

SERIES_FIELDS = ['t', 'ibvs_err_fresh', 'converged', 'feat_valid', 'feat_source',
                 'u', 'v', 'area', 'angle', 'e_x', 'e_y', 'e_area', 'e_angle',
                 'v_x', 'v_y', 'v_z', 'w_x', 'w_y', 'w_z',
                 'gt_ex', 'gt_ey', 'gt_ez', 'gt_rx', 'gt_ry', 'gt_rz', 'gt_et', 'gt_er',
                 'cam_x', 'cam_y', 'cam_z', 'cam_qx', 'cam_qy', 'cam_qz', 'cam_qw']

SUMMARY_FIELDS = ['target', 'start', 'detector', 'tracker', 'detector_period', 'detector_delay',
                  'occlusion', 'light', 'success', 'converged', 'conv_time_s', 'outcome',
                  'final_pos_err_m', 'final_rot_err_deg', 'final_rot_err_raw_deg',
                  'final_feature_err', 'detector_calls', 'lost_stop_time_s', 'drift_after_loss_m']


def symmetric_rot_err(rx, ry, rz, symmetry):
    """Rotation error that ignores what the target's symmetry makes invisible (small-tilt approx.)."""
    if symmetry == 'round':
        return math.hypot(rx, ry)
    if symmetry == 'pi':
        rz = (rz + math.pi / 2) % math.pi - math.pi / 2
    return math.sqrt(rx * rx + ry * ry + rz * rz)


def gz_service(service, reqtype, req, attempts=3):
    """Call a service of the running Gazebo world (GZ_IP / GZ_PARTITION come from the env)."""
    for _ in range(attempts):  # light_config occasionally times out on the first call
        res = subprocess.run(['gz', 'service', '-s', f'/world/vs_world/{service}',
                              '--reqtype', reqtype, '--reptype', 'gz.msgs.Boolean',
                              '--timeout', '5000', '--req', req], capture_output=True, text=True)
        if 'data: true' in res.stdout:
            return
    raise RuntimeError(f'gz service {service} failed: {res.stdout or res.stderr}')


def set_light(factor):
    for name, spec in LIGHTS.items():
        gz_service('light_config', 'gz.msgs.Light', f'name: "{name}", {spec}, intensity: {factor}')


def spawn_occluder(target_cfg, fraction):
    """Grey plate above the target that covers `fraction` of its footprint from the +x side."""
    x, y, z, _ = target_cfg['pose']
    fx = target_cfg['footprint'][0]
    depth = 0.15
    cx = x + fx / 2 - fraction * fx + depth / 2
    sdf = (f'<sdf version="1.9"><model name="{OCCLUDER}"><static>true</static><link name="l">'
           f'<visual name="v"><geometry><box><size>{depth} 0.25 0.004</size></box></geometry>'
           f'<material><diffuse>0.35 0.35 0.38 1</diffuse><ambient>0.35 0.35 0.38 1</ambient>'
           f'</material></visual></link></model></sdf>')
    req = (f'sdf: {json.dumps(sdf)}, pose: {{position: {{x: {cx}, y: {y}, '
           f'z: {z + target_cfg["top"] + OCCLUDER_HEIGHT}}}}}')
    gz_service('create', 'gz.msgs.EntityFactory', req)


def remove_occluder():
    try:
        gz_service('remove', 'gz.msgs.Entity', f'name: "{OCCLUDER}", type: MODEL')
    except RuntimeError:
        pass  # not there


def restore(args):
    """Undo the disturbances of this run (only what was changed)."""
    if args.occlusion > 0:
        remove_occluder()
    if args.light != 1.0:
        set_light(1.0)


def move_model(name, x, y, z, yaw=0.0):
    gz_service('set_pose', 'gz.msgs.Pose',
               f'name: "{name}", position: {{x: {x}, y: {y}, z: {z}}}, '
               f'orientation: {{z: {math.sin(yaw / 2)}, w: {math.cos(yaw / 2)}}}')


class Recorder(Node):
    """Keeps the latest value of every topic; samples one row at 20 Hz while recording."""

    def __init__(self):
        super().__init__('vs_eval_recorder')
        self.feat = self.gt = self.cam = self.twist = self.err = self.image = self.K = None
        self.target_pose = None
        self.err_fresh = False
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
        self.create_subscription(PoseStamped, '/ground_truth/target_pose',
                                 lambda m: setattr(self, 'target_pose', m.pose), 10)
        self.create_subscription(TwistStamped, '/ibvs_node/camera_twist',
                                 lambda m: setattr(self, 'twist', m.twist), 10)
        self.create_subscription(Bool, '/ibvs_node/converged',
                                 lambda m: setattr(self, 'converged', m.data), 10)
        self.create_subscription(Float64MultiArray, '/ibvs_node/error', self.on_error, 10)
        self.create_subscription(Image, '/camera/image_raw',
                                 lambda m: setattr(self, 'image', m), qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/camera/camera_info',
                                 lambda m: setattr(self, 'K', list(m.k)), qos_profile_sensor_data)
        self.create_subscription(Clock, '/clock',
                                 lambda _: setattr(self, 'clock_seen', time.monotonic()),
                                 qos_profile_sensor_data)
        self.create_timer(1.0 / SAMPLE_RATE, self.sample)
        self.cli = {n: self.create_client(Trigger, f'/ibvs_node/{n}')
                    for n in ('start', 'stop', 'capture_reference')}
        self.set_cli = self.create_client(SetParameters, '/perception_node/set_parameters')
        self.ibvs_set_cli = self.create_client(SetParameters, '/ibvs_node/set_parameters')
        self.get_cli = self.create_client(GetParameters, '/perception_node/get_parameters')

    def on_feat(self, m):
        self.feat = m
        if m.source == 'detector':
            self.detections += 1

    def on_error(self, m):
        self.err, self.err_fresh = list(m.data), True

    def sample(self):
        if self.rows is None:
            return
        nan = math.nan
        e = (self.err or []) + [nan] * (4 - len(self.err or []))
        f, g, c, w = self.feat, self.gt or [nan] * 8, self.cam, self.twist
        row = [time.monotonic() - self.t0, int(self.err_fresh), int(self.converged),
               int(bool(f and f.valid)), f.source if f else '',
               f.u if f else nan, f.v if f else nan, f.area if f else nan, f.angle if f else nan,
               *e[:4]]
        row += ([w.linear.x, w.linear.y, w.linear.z, w.angular.x, w.angular.y, w.angular.z]
                if w else [nan] * 6)
        row += g[:8]
        row += ([c.position.x, c.position.y, c.position.z, c.orientation.x, c.orientation.y,
                 c.orientation.z, c.orientation.w] if c else [nan] * 7)
        self.rows.append(row)
        self.err_fresh = False

    def spin_for(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.02)

    def check_sim_clock(self):
        # A stalled /clock freezes the trajectory controller: goals are accepted, never executed
        self.spin_for(1.0)
        if time.monotonic() - self.clock_seen > 1.0:
            raise RuntimeError('no /clock from the simulation: restart the sim container')

    def call(self, name, timeout=10.0):
        cli = self.cli[name]
        if not cli.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f'/ibvs_node/{name} not available')
        fut = cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        return fut.result()

    def set_param(self, name, value, node='perception'):
        cli = self.set_cli if node == 'perception' else self.ibvs_set_cli
        if not cli.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(f'{node} set_parameters not available')
        pv = (ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=value)
              if isinstance(value, str) else
              ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(value)))
        fut = cli.call_async(SetParameters.Request(parameters=[Parameter(name=name, value=pv)]))
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)

    def perception_params(self):
        names = ['detector', 'tracker', 'detector_period', 'detector_delay']
        self.get_cli.wait_for_service(timeout_sec=10.0)
        fut = self.get_cli.call_async(GetParameters.Request(names=names))
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        out = {}
        for n, v in zip(names, fut.result().values):
            out[n] = v.string_value if v.type == ParameterType.PARAMETER_STRING else v.double_value
        return out

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

    def save_view(self, folder, name, target):
        """Camera frame plus ground truth for the offline prompt / detector comparison."""
        self.image = None
        self.spin_for(0.5)
        m = self.image
        if m is None or self.cam is None or self.target_pose is None:
            return
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
        cv2.imwrite(os.path.join(folder, name + '.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        def pose(p):
            return [p.position.x, p.position.y, p.position.z,
                    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
        with open(os.path.join(folder, name + '.json'), 'w') as fh:
            json.dump({'target': target, 'K': self.K, 'camera_pose_world': pose(self.cam),
                       'target_pose_world': pose(self.target_pose)}, fh, indent=1)


def stop_metrics(rows, lost_t):
    """Time until the camera stays within 1 mm of its final position, and total drift after loss."""
    after = [r for r in rows if r[0] >= lost_t and not math.isnan(r[27])]
    if len(after) < 2:
        return '', ''
    pos = np.array([r[27:30] for r in after])
    drift = float(np.linalg.norm(pos[-1] - pos[0]))
    moving = np.linalg.norm(pos - pos[-1], axis=1) > 0.001
    idx = int(np.max(np.nonzero(moving))) + 1 if moving.any() else 0
    return round(after[min(idx, len(after) - 1)][0] - lost_t, 2), round(drift, 5)


def run_one(rec, symmetry, args, target_cfg):
    """Servo from the current pose. Returns the summary values and the recorded rows."""
    rec.spin_for(0.3)  # drop converged messages still queued from the previous run
    rec.rows, rec.t0, rec.converged, rec.detections = [], time.monotonic(), False, 0
    rec.call('start')
    conv_time = lost_t = None
    while time.monotonic() - rec.t0 < args.timeout:
        rclpy.spin_once(rec, timeout_sec=0.02)
        elapsed = time.monotonic() - rec.t0
        if args.lose_target_after is not None:
            if lost_t is None and elapsed >= args.lose_target_after:
                move_model(target_cfg['model'], 3.0, 3.0, 0.0)   # far out of view
                lost_t = elapsed
            if lost_t is not None and elapsed > lost_t + 5.0:
                break
            continue
        if rec.converged:
            conv_time = elapsed
            break
        if rec.feat is not None and not rec.feat.valid and elapsed > 2.0:
            break
    rec.spin_for(0.5)  # a few rows in the final pose
    rec.call('stop')
    rows, rec.rows = rec.rows, None
    result = {'lost_stop_time_s': '', 'drift_after_loss_m': ''}
    if lost_t is not None:
        x, y, z, yaw = target_cfg['pose']
        move_model(target_cfg['model'], x, y, z, yaw)
        stop_t, drift = stop_metrics(rows, lost_t)
        result.update(outcome='lost test', converged=0, conv_time_s='', lost_stop_time_s=stop_t,
                      drift_after_loss_m=drift,
                      success=int(stop_t != '' and stop_t < 1.0 and drift < 0.005))
        return result, rows
    g = rec.gt or [math.nan] * 8
    last = rows[-1] if rows else None
    feat_err = (math.sqrt(sum(x * x for x in last[9:13] if not math.isnan(x))) if last else math.nan)
    if conv_time is not None:
        outcome = 'converged'
    elif rec.feat is not None and not rec.feat.valid:
        outcome = 'target lost'
    else:
        outcome = 'timeout'
    rot = math.degrees(symmetric_rot_err(g[3], g[4], g[5], symmetry))
    result.update(converged=int(conv_time is not None),
                  conv_time_s=round(conv_time, 2) if conv_time is not None else '',
                  outcome=outcome, final_pos_err_m=round(g[6], 5), final_rot_err_deg=round(rot, 3),
                  final_rot_err_raw_deg=round(math.degrees(g[7]), 3),
                  final_feature_err=round(feat_err, 5), detector_calls=rec.detections)
    result['success'] = int(conv_time is not None and g[6] < args.pos_tol and rot < args.rot_tol)
    return result, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--targets', nargs='+', default=list(TARGETS), choices=list(TARGETS))
    ap.add_argument('--starts', nargs='+', type=int, default=list(range(1, 9)))
    ap.add_argument('--timeout', type=float, default=60.0, help='seconds per run')
    ap.add_argument('--pos-tol', type=float, default=0.005, help='success tolerance [m]')
    ap.add_argument('--rot-tol', type=float, default=2.0, help='success tolerance [deg]')
    ap.add_argument('--occlusion', type=float, default=0.0,
                    help='fraction of the target footprint covered by a plate (nominal, 0-1)')
    ap.add_argument('--light', type=float, default=1.0, help='intensity factor for both lights')
    ap.add_argument('--detector-period', type=float, help='seconds between FM calls')
    ap.add_argument('--detector-delay', type=float,
                    help='extra seconds added to every FM call, emulates slower hardware')
    ap.add_argument('--feature-timeout', type=float,
                    help='max. feature age for the controller [s]; above the detector period '
                         'for runs without tracker, else the run ends at the first gap')
    ap.add_argument('--lose-target-after', type=float,
                    help='move the target out of view after this many seconds (safe stop test)')
    ap.add_argument('--label', default='run', help='name part of the output folder')
    ap.add_argument('--out', default='/root/vs_data/eval')
    args = ap.parse_args()

    out = os.path.join(args.out, time.strftime('%Y%m%d_%H%M%S') + '_' + args.label)
    for sub in ('runs', 'images'):
        os.makedirs(os.path.join(out, sub))

    rclpy.init()
    mover, rec = RobotMover(), Recorder()
    mover.ready()
    if args.detector_period is not None:
        rec.set_param('detector_period', args.detector_period)
    if args.detector_delay is not None:
        rec.set_param('detector_delay', args.detector_delay)
    if args.feature_timeout is not None:
        rec.set_param('feature_timeout', args.feature_timeout, node='ibvs')
    params = rec.perception_params()
    with open(os.path.join(out, 'config.json'), 'w') as fh:
        json.dump({**vars(args), 'perception': params}, fh, indent=2)
    fixed = {'detector': params['detector'], 'tracker': params['tracker'],
             'detector_period': params['detector_period'],
             'detector_delay': params['detector_delay'],
             'occlusion': args.occlusion, 'light': args.light}

    try:
        with open(os.path.join(out, 'summary.csv'), 'w', newline='') as fh:
            summary = csv.DictWriter(fh, SUMMARY_FIELDS)
            summary.writeheader()
            for target in args.targets:
                prompt, symmetry = TARGETS[target]
                target_cfg = mover.cfg['targets'][target]
                rec.call('stop')
                rec.set_param('prompt', prompt)
                rec.check_sim_clock()
                mover.go_to(target, start=None)
                rec.save_view(os.path.join(out, 'images'), f'{target}_reference', target)
                if (not rec.wait_target(prompt, 30.0)
                        or not rec.call('capture_reference').success):
                    print(f'{target}: no detection at the reference pose, skipped', flush=True)
                    continue
                if args.occlusion > 0:
                    spawn_occluder(target_cfg, args.occlusion)
                if args.light != 1.0:
                    set_light(args.light)
                for start in args.starts:
                    rec.check_sim_clock()
                    mover.go_to(target, start=start)
                    rec.save_view(os.path.join(out, 'images'), f'{target}_S{start}', target)
                    row = {'target': target, 'start': start, **fixed}
                    if not rec.wait_target(prompt, 20.0):
                        row.update(success=0, converged=0, outcome='not detected')
                        rows = []
                    else:
                        result, rows = run_one(rec, symmetry, args, target_cfg)
                        row.update(result)
                    summary.writerow(row)
                    fh.flush()
                    with open(os.path.join(out, 'runs', f'{target}_S{start}.csv'), 'w',
                              newline='') as rf:
                        w = csv.writer(rf)
                        w.writerow(SERIES_FIELDS)
                        w.writerows(rows)
                    print(f'{target} S{start}: {row.get("outcome")} success={row["success"]} '
                          f'pos={row.get("final_pos_err_m", "")} '
                          f'rot={row.get("final_rot_err_deg", "")} t={row.get("conv_time_s", "")} '
                          f'stop={row.get("lost_stop_time_s", "")}', flush=True)
                restore(args)
    finally:
        rec.call('stop')
        restore(args)
    print(f'results in {out}', flush=True)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
