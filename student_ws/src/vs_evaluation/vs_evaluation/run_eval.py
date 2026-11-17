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
import threading
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
from std_msgs.msg import Bool, Float64MultiArray, String
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
    # Full spec incl. pose as in worlds/vs_world.sdf: fields left out may be reset by Gazebo
    'sun': 'type: DIRECTIONAL, pose: {position: {z: 10}}, diffuse: {r: 0.85, g: 0.85, b: 0.85, a: 1}, '
           'specular: {r: 0.2, g: 0.2, b: 0.2, a: 1}, direction: {x: -0.3, y: 0.2, z: -0.9}, '
           'cast_shadows: true',
    'lab_light': 'type: POINT, pose: {position: {x: 0.6, y: -0.4, z: 1.6}}, '
                 'diffuse: {r: 0.4, g: 0.4, b: 0.4, a: 1}, '
                 'specular: {r: 0.1, g: 0.1, b: 0.1, a: 1}, range: 6, attenuation_constant: 0.6, '
                 'attenuation_linear: 0.05, attenuation_quadratic: 0.01',
}
OCCLUDER = 'vs_occluder'
# Just above the target top: a plate higher up (closer to the camera) covered far more than the
# nominal fraction from the shifted start poses through parallax
OCCLUDER_HEIGHT = 0.01
SAMPLE_RATE = 20.0

SERIES_FIELDS = ['t', 'ibvs_err_fresh', 'converged', 'feat_valid', 'feat_source',
                 'u', 'v', 'area', 'angle', 'e_x', 'e_y', 'e_area', 'e_angle',
                 'v_x', 'v_y', 'v_z', 'w_x', 'w_y', 'w_z',
                 'gt_ex', 'gt_ey', 'gt_ez', 'gt_rx', 'gt_ry', 'gt_rz', 'gt_et', 'gt_er',
                 'cam_x', 'cam_y', 'cam_z', 'cam_qx', 'cam_qy', 'cam_qz', 'cam_qw', 'sim_t']

SUMMARY_FIELDS = ['target', 'start', 'detector', 'tracker', 'detector_period', 'detector_delay',
                  'occlusion', 'occlusion_after', 'light', 'success', 'converged', 'conv_time_s', 'outcome',
                  'final_pos_err_m', 'final_rot_err_deg', 'final_rot_err_raw_deg',
                  'final_feature_err', 'detector_calls', 'lost_stop_time_s', 'drift_after_loss_m',
                  'conv_time_sim_s', 'real_time_factor', 'stop_reason', 'detector_model']


def _rotvec_to_matrix(r):
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3)
    k = np.asarray(r) / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(theta) * kx + (1 - math.cos(theta)) * kx @ kx


def _angle(R):
    return math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))


def symmetric_rot_err(rx, ry, rz, symmetry):
    """Rotation error (rad) that ignores what the target's symmetry makes invisible.

    (rx, ry, rz) is the rotation vector of R_d^T R_c in the desired camera frame (lab ground
    truth). Exact angles: the earlier small-tilt version overstated the error at large yaw.
    """
    if any(math.isnan(v) for v in (rx, ry, rz)):
        return math.nan
    R = _rotvec_to_matrix((rx, ry, rz))
    if symmetry == 'round':  # yaw invisible: angle between the optical axes only
        return math.acos(max(-1.0, min(1.0, R[2, 2])))
    if symmetry == 'pi':  # the box looks the same after half a turn about the optical axis
        return min(_angle(R), _angle(np.diag([-1.0, -1.0, 1.0]) @ R))
    return _angle(R)


def stop_reason_label(status):
    """Outcome label from the controller's last halt reason (ibvs_node ~/status)."""
    s = status.lower()
    if any(k in s for k in ('collision', 'below', 'workspace')):
        return 'safety stop'
    if s.startswith('tf'):
        return 'tf error'
    if 'stale' in s:
        return 'features stale'
    if 'lost' in s:
        return 'target lost'
    return 'controller stopped'



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
    fx, fy = target_cfg['footprint']
    # Sized to the target: a large board also hid the neighbouring objects
    depth, width = 0.08, fy + 0.04
    cx = x + fx / 2 - fraction * fx + depth / 2
    sdf = (f'<sdf version="1.9"><model name="{OCCLUDER}"><static>true</static><link name="l">'
           f'<visual name="v"><geometry><box><size>{depth} {width} 0.004</size></box></geometry>'
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
        self.err_time = 0.0
        self.converged = False
        self.clock_seen = 0.0
        self.sim_now = math.nan
        self.status = ''
        self.hold_time = 1.0  # read from the controller in main()
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
        self.create_subscription(Clock, '/clock', self.on_clock, qos_profile_sensor_data)
        self.create_subscription(String, '/ibvs_node/status',
                                 lambda m: setattr(self, 'status', m.data), 10)
        self.create_timer(1.0 / SAMPLE_RATE, self.sample)
        self.cli = {n: self.create_client(Trigger, f'/ibvs_node/{n}')
                    for n in ('start', 'stop', 'capture_reference')}
        self.set_cli = self.create_client(SetParameters, '/perception_node/set_parameters')
        self.ibvs_set_cli = self.create_client(SetParameters, '/ibvs_node/set_parameters')
        self.get_cli = self.create_client(GetParameters, '/perception_node/get_parameters')
        self.ibvs_get_cli = self.create_client(GetParameters, '/ibvs_node/get_parameters')

    def on_feat(self, m):
        self.feat = m
        if m.source == 'detector':
            self.detections += 1

    def on_clock(self, m):
        self.clock_seen = time.monotonic()
        self.sim_now = m.clock.sec + m.clock.nanosec * 1e-9

    def on_error(self, m):
        self.err, self.err_fresh, self.err_time = list(m.data), True, time.monotonic()

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
        row.append(self.sim_now)
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
        if not cli.wait_for_service(timeout_sec=60.0):  # perception may be respawning
            raise RuntimeError(f'{node} set_parameters not available')
        pv = (ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=value)
              if isinstance(value, str) else
              ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(value)))
        fut = cli.call_async(SetParameters.Request(parameters=[Parameter(name=name, value=pv)]))
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)

    def ibvs_param(self, name, default):
        if not self.ibvs_get_cli.wait_for_service(timeout_sec=10.0):
            return default
        fut = self.ibvs_get_cli.call_async(GetParameters.Request(names=[name]))
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        values = fut.result().values if fut.result() else []
        return values[0].double_value if values and values[0].type == 3 else default

    def perception_params(self):
        names = ['detector', 'tracker', 'detector_period', 'detector_delay', 'detector_model']
        self.get_cli.wait_for_service(timeout_sec=10.0)
        fut = self.get_cli.call_async(GetParameters.Request(names=names))
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        out = {}
        for n, v in zip(names, fut.result().values):
            out[n] = v.string_value if v.type == ParameterType.PARAMETER_STRING else v.double_value
        return out

    def wait_target(self, prompt, timeout, hold=1.0, max_gap=2.0):
        """Wait until the target with this prompt has been seen for `hold` seconds.

        Detector or tracker, both count: with detect_on_loss the detector only runs while no
        track exists, so after a move the tracker may be the only source. Gaps up to `max_gap`
        are allowed: without a tracker, features are only valid on detector frames.
        """
        self.feat, first, last, n = None, None, None, 0
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            f, now = self.feat, time.monotonic()
            self.feat = None  # count every message once
            if f and f.valid and f.prompt == prompt:
                if last is None or now - last > max_gap:
                    first, n = now, 0
                last, n = now, n + 1
                if now - first >= hold and n >= 2:
                    return True
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
    # Per-run state: values left from the previous run gave wrong final errors and labels
    rec.err, rec.err_fresh, rec.status = None, False, ''
    rec.rows, rec.t0, rec.converged, rec.detections = [], time.monotonic(), False, 0
    rec.err_time = rec.t0
    sim0 = rec.sim_now
    rec.call('start')
    conv_time = conv_sim = lost_t = mover = occluder = None
    outcome = 'timeout'
    moved = threading.Event()
    while time.monotonic() - rec.t0 < args.timeout:
        rclpy.spin_once(rec, timeout_sec=0.02)
        elapsed = time.monotonic() - rec.t0
        if args.occlusion_after is not None and occluder is None and elapsed >= args.occlusion_after:
            occluder = threading.Thread(target=spawn_occluder,
                                        args=(target_cfg, args.occlusion), daemon=True)
            occluder.start()
        if args.lose_target_after is not None:
            # The gz service call takes seconds; in a thread the recorder keeps sampling
            if mover is None and elapsed >= args.lose_target_after:
                def lose():
                    move_model(target_cfg['model'], 3.0, 3.0, 0.0)   # far out of view
                    moved.set()
                mover = threading.Thread(target=lose, daemon=True)
                mover.start()
                # Loss time = when the move is issued: the earliest the target can vanish, so
                # stop time and drift are never understated (the end of the service call came
                # later; the lab's ground truth does not see the moved model at all)
                lost_t = elapsed
            if lost_t is not None and elapsed > lost_t + 5.0:
                break
            continue
        if rec.converged:
            # The controller reports convergence after holding the error below its threshold
            # for converged_hold_time; the convergence time is when that hold began
            conv_time, outcome = max(0.0, elapsed - rec.hold_time), 'converged'
            conv_sim = rec.sim_now - sim0 - rec.hold_time
            break
        # The controller decides about a lost target (lost_timeout, stale features); its error
        # stream stopping is the sign. A stricter check here ended runs of slow detectors at
        # every short tracker gap the controller rides out.
        if elapsed > 2.0 and time.monotonic() - rec.err_time > 2.0:
            outcome = stop_reason_label(rec.status)
            break
    rec.spin_for(0.5)  # a few rows in the final pose
    rec.call('stop')
    if occluder is not None:
        occluder.join()
        remove_occluder()  # the next start detects without it
    rows, rec.rows = rec.rows, None
    result = {'lost_stop_time_s': '', 'drift_after_loss_m': ''}
    if mover is not None:
        mover.join()
        lost_t = lost_t if lost_t is not None else elapsed
        x, y, z, yaw = target_cfg['pose']
        move_model(target_cfg['model'], x, y, z, yaw)
        stop_t, drift = stop_metrics(rows, lost_t)
        result.update(outcome='lost test', converged=0, conv_time_s='', lost_stop_time_s=stop_t,
                      drift_after_loss_m=drift,
                      success=int(stop_t != '' and stop_t < 1.0 and drift < 0.005))
        return result, rows
    g = rec.gt or [math.nan] * 8
    wall = time.monotonic() - rec.t0
    rtf = (rec.sim_now - sim0) / wall if wall > 0 and not math.isnan(sim0) else math.nan
    result.update(conv_time_sim_s=round(conv_sim, 2) if conv_sim is not None else '',
                  real_time_factor=round(rtf, 3), stop_reason=rec.status)
    last = rows[-1] if rows else None
    feat_err = (math.sqrt(sum(x * x for x in last[9:13] if not math.isnan(x))) if last else math.nan)
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
    ap.add_argument('--occlusion-after', type=float,
                    help='spawn the occluder this many seconds into each run instead of before it '
                         '(tests the tracker, not the detection under occlusion)')
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
    ap.add_argument('--resume', help='output folder of an interrupted run: append, skip done runs')
    ap.add_argument('--out', default='/root/vs_data/eval')
    args = ap.parse_args()

    out = args.resume or os.path.join(args.out, time.strftime('%Y%m%d_%H%M%S') + '_' + args.label)
    for sub in ('runs', 'images'):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    summary_path = os.path.join(out, 'summary.csv')
    done = set()
    if os.path.exists(summary_path):
        with open(summary_path) as fh:
            done = {(r['target'], int(r['start'])) for r in csv.DictReader(fh)}

    rclpy.init()
    mover, rec = RobotMover(), Recorder()
    mover.ready()
    rec.hold_time = rec.ibvs_param('converged_hold_time', 1.0)

    def configure_perception(prompt=None):
        # Re-applied before every run: a respawned perception node starts with its defaults
        if prompt is not None:
            rec.set_param('prompt', prompt)
        if args.detector_period is not None:
            rec.set_param('detector_period', args.detector_period)
        if args.detector_delay is not None:
            rec.set_param('detector_delay', args.detector_delay)

    configure_perception()
    if args.feature_timeout is not None:
        rec.set_param('feature_timeout', args.feature_timeout, node='ibvs')
    params = rec.perception_params()
    with open(os.path.join(out, 'config.json'), 'w') as fh:
        json.dump({**vars(args), 'perception': params}, fh, indent=2)
    fixed = {'detector': params['detector'], 'detector_model': params.get('detector_model', ''),
             'tracker': params['tracker'],
             'detector_period': params['detector_period'],
             'detector_delay': params['detector_delay'],
             'occlusion': args.occlusion, 'occlusion_after': args.occlusion_after,
             'light': args.light}

    try:
        with open(summary_path, 'a', newline='') as fh:
            summary = csv.DictWriter(fh, SUMMARY_FIELDS)
            if fh.tell() == 0:  # also after an attempt that crashed before its first run
                summary.writeheader()
            for target in args.targets:
                todo = [s for s in args.starts if (target, s) not in done]
                if not todo:
                    continue
                prompt, symmetry = TARGETS[target]
                target_cfg = mover.cfg['targets'][target]
                rec.call('stop')
                rec.check_sim_clock()
                mover.go_to(target, start=None)
                # Prompt only after the move: detecting in the old view picked another object
                # (green cup -> yellow ball) and the tracker kept it all the way
                configure_perception(prompt)
                rec.save_view(os.path.join(out, 'images'), f'{target}_reference', target)
                if (not rec.wait_target(prompt, 30.0)
                        or not rec.call('capture_reference').success):
                    # Counted as failures, otherwise a weak detector looks better than it is
                    print(f'{target}: no detection at the reference pose', flush=True)
                    for start in todo:
                        summary.writerow({'target': target, 'start': start, **fixed,
                                          'success': 0, 'converged': 0,
                                          'outcome': 'no reference'})
                    fh.flush()
                    continue
                if args.occlusion > 0 and args.occlusion_after is None:
                    spawn_occluder(target_cfg, args.occlusion)
                if args.light != 1.0:
                    set_light(args.light)
                for start in todo:
                    rec.check_sim_clock()
                    mover.go_to(target, start=start)
                    configure_perception(prompt)
                    rec.save_view(os.path.join(out, 'images'), f'{target}_S{start}', target)
                    row = {'target': target, 'start': start, **fixed}
                    if not rec.wait_target(prompt, 40.0):
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
