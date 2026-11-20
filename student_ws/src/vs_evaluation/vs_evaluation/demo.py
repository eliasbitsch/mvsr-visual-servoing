"""Step-by-step pipeline for the lab demo (marimo app notebooks/demo.py, also usable from
Jupyter), each step followed by its measurement. Reuses the evaluation runner's recorder and Gazebo helpers, so the numbers are the
same ones the evaluation reports.
"""
import io
import math
import subprocess
import threading
from collections import deque
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage
from vs_sim.motion import RobotMover

from vs_evaluation.run_eval import (SERIES_FIELDS, TARGETS, Recorder, move_model,
                                    remove_occluder, set_light, spawn_occluder, stop_metrics,
                                    symmetric_rot_err)

POS_TOL, ROT_TOL = 0.005, 2.0
# Perception frames per second (SAM 2 only as often as needed): the controller runs at 30 Hz on
# the newest features, 5 Hz were enough to converge; standing still needs even less
IDLE_RATE, SERVO_RATE = 2.0, 5.0


def _marimo():
    try:
        import marimo as mo
    except ImportError:
        return None
    return mo if mo.running_in_notebook() else None


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=90)
    return buf.getvalue()


def _show_image(data, width):
    mo = _marimo()
    if mo:
        mo.output.append(mo.image(src=data, width=width))
    else:
        from IPython.display import Image, display
        display(Image(data=data, width=width))


def _show_fig(fig):
    mo = _marimo()
    if mo:
        mo.output.append(mo.image(src=_png(fig)))
    else:
        from IPython.display import display
        display(fig)
    plt.close(fig)


class _Live:
    """A figure redrawn in place while the robot moves (marimo: replaces the cell output)."""

    def __init__(self, fig, extra=None):
        self.fig, self.mo, self.extra = fig, _marimo(), extra
        if not self.mo:
            from IPython.display import display
            self.handle = display(fig, display_id=True)
        plt.close(fig)

    def update(self):
        if self.mo:
            parts = [self.extra()] if self.extra else []
            self.mo.output.replace(self.mo.vstack(parts + [self.mo.image(src=_png(self.fig))]))
        elif self.handle is not None:  # None outside a notebook (headless test)
            self.handle.update(self.fig)


def _rate(stamps):
    """Rate from message stamps: right even when callbacks run in bursts (between cells)."""
    if len(stamps) < 2 or time.monotonic() - stamps[-1][1] > 2.0:
        return 0.0
    # Median gap: messages dropped between two bursts would stretch a plain span
    gaps = [b[0] - a[0] for a, b in zip(stamps, list(stamps)[1:]) if b[0] > a[0]]
    return 1.0 / float(np.median(gaps)) if gaps else 0.0


def _stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, time.monotonic()


class Demo(Recorder):

    def __init__(self):
        if not rclpy.ok():
            rclpy.init()
        super().__init__()
        self.debug_jpeg = None
        # The perception node encodes this only while someone subscribes
        self.create_subscription(CompressedImage, '/vs/debug_image/compressed',
                                 lambda m: setattr(self, 'debug_jpeg', bytes(m.data)), 2)
        # Arrival times for the live pipeline view
        # (header stamp, arrival) pairs; the stamps give the rate
        self.cam_times, self.feat_times = deque(maxlen=20), deque(maxlen=20)
        self.det_time = 0.0
        # camera_info comes with every frame and is tiny; the raw image would cost 9 MB/s here
        self.create_subscription(CameraInfo, '/camera/camera_info',
                                 lambda m: self.cam_times.append(_stamp(m)),
                                 qos_profile_sensor_data)
        self.mover = RobotMover(name='vs_demo_mover')
        self.mover.ready()
        self.check_sim_clock()
        self.target = None
        self.text = None  # prompt currently used for servoing
        self.last = None
        self.history = []  # one row per servo run of this session
        self.start_name = None
        self.disturbance = 'none'
        # The evaluation uses 0.5 s / 2 s; on a throttling laptop GPU the perception sometimes
        # pauses for a second, and the controller would end a demo run there
        self.set_param('feature_timeout', 1.5, node='ibvs')
        self.set_param('lost_timeout', 3.0, node='ibvs')
        self.set_param('max_rate', IDLE_RATE)
        p = self.params = self.perception_params()
        print(f'connected: detector {p["detector"]}, tracker {p["tracker"]} '
              f'(demo tolerances: feature timeout 1.5 s, lost after 3 s)')

    def on_feat(self, m):
        super().on_feat(m)
        self.feat_times.append(_stamp(m))
        if m.source == 'detector':
            self.det_time = time.monotonic()

    # --- views ----------------------------------------------------------------------------

    def flow(self):
        """Pipeline as a flowchart (marimo) with live values; tinted stages are working now."""
        import marimo as mo

        from vs_evaluation.flow_view import mermaid
        now, f, w = time.monotonic(), self.feat, self.twist
        valid = bool(f and f.valid)
        speed = math.sqrt(w.linear.x ** 2 + w.linear.y ** 2 + w.linear.z ** 2) if w else 0.0
        err = math.sqrt(sum(x * x for x in (self.err or [])[:4] if not math.isnan(x)))
        controlling = now - self.err_time < 0.6
        moving = controlling and speed > 1e-3
        cam_hz, feat_hz = _rate(self.cam_times), _rate(self.feat_times)
        tracking = valid and f.source == 'tracker'
        state = {
            'prompt': self.text or '-',
            'cam': f'{cam_hz:.1f} Hz',
            'detector': self.params.get('detector', '?'),
            'latency': f'{f.detector_latency * 1000:.0f} ms per call' if f else '-',
            'tracker': self.params.get('tracker', '?'),
            'features': f'{feat_hz:.1f} Hz, area {f.area:.0f} px2' if valid else 'no target',
            'error': f'feature error {err:.3f}' if controlling else 'idle',
            'speed': f'{speed * 1000:.0f} mm/s',
            'on': {'P': bool(self.text), 'CAM': cam_hz > 0.5, 'D1': valid,
                   'FM': now - self.det_time < 1.5, 'TR': tracking, 'FE': valid,
                   'IB': controlling, 'D2': controlling, 'KI': moving, 'RO': moving,
                   'HO': bool(self.converged) and not moving},
        }
        return mo.mermaid(mermaid(state))

    def show(self, width=520):
        """Camera image with the detection (green: detector, orange: tracker) and the features."""
        self.debug_jpeg = None
        self.spin_for(0.6)
        if self.debug_jpeg:
            _show_image(self.debug_jpeg, width)
        f = self.feat
        if f and f.valid:
            axis = f'{math.degrees(f.angle):.1f} deg' if f.angle_valid else 'none (round)'
            print(f'{f.source}: centre u={f.u:.1f} v={f.v:.1f} px | area {f.area:.0f} px2 | '
                  f'axis {axis} | score {f.score:.2f} | detector latency '
                  f'{f.detector_latency * 1000:.0f} ms')
        else:
            print('no valid target in view')

    def pose_error(self):
        """Ground-truth camera pose error to the reference (evaluation only, not used in control)."""
        self.spin_for(0.3)
        g = self.gt
        if not g or self.target is None:
            return None
        rot = math.degrees(symmetric_rot_err(g[3], g[4], g[5], TARGETS[self.target][1]))
        print(f'pose error: position {g[6] * 1000:.1f} mm, rotation {rot:.2f} deg '
              f'(symmetry-aware), raw {math.degrees(g[7]):.1f} deg')
        return g[6], rot

    # --- pipeline steps -------------------------------------------------------------------

    def prompt(self, text, timeout=15.0, show=True):
        """Any prompt in the current view: is something found, how fast, where. Servoing keeps
        its own prompt (set again afterwards); use reference(target, prompt=...) to servo on it."""
        t0 = time.monotonic()
        self.set_param('prompt', text)
        found = self.wait_target(text, timeout)
        dt = time.monotonic() - t0
        if show:
            print(f'"{text}": ' + (f'found after {dt:.1f} s' if found
                                   else f'nothing found within {timeout:.0f} s'))
            self.show()
            self._restore_prompt()
        return found, dt

    def _restore_prompt(self):
        if self.text:
            self.set_param('prompt', self.text)
            self.wait_target(self.text, 20.0)  # before a capture_reference right after

    def compare_prompts(self, prompts, timeout=6.0, cols=3):
        """Several formulations in the same view, one image per prompt. The servo prompt is
        set again afterwards."""
        from PIL import Image as PILImage
        shots = []
        for text in prompts:
            found, dt = self.prompt(text, timeout, show=False)
            self.debug_jpeg = None
            self.spin_for(0.6)
            f = self.feat
            shots.append((text, found, dt, self.debug_jpeg,
                          f.score if found and f and f.valid else None))
        rows = math.ceil(len(shots) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.7 * rows), squeeze=False)
        for ax in axes.flat:
            ax.axis('off')
        for ax, (text, found, dt, jpeg, score) in zip(axes.flat, shots):
            if jpeg:
                ax.imshow(PILImage.open(io.BytesIO(jpeg)))
            label = 'not found' if not found else f'found in {dt:.1f} s' + (
                f', score {score:.2f}' if score is not None else '')
            ax.set_title(f'"{text}"\n{label}', fontsize=10,
                         color='black' if found else 'tab:red')
        fig.tight_layout()
        _show_fig(fig)
        self._restore_prompt()
        return pd.DataFrame([{'prompt': t, 'found': f, 'time [s]': round(d, 1),
                              'score': None if s is None else round(s, 2)}
                             for t, f, d, _, s in shots])

    def reference(self, target, prompt=None):
        """Drive to the reference pose and let the foundation model find the target.

        prompt: own formulation instead of the default one (e.g. "red soda tin").
        """
        self.target = target
        prompt = self.text = prompt or TARGETS[target][0]
        self.call('stop')
        self.mover.go_to(target, start=None)
        t0 = time.monotonic()
        self.set_param('prompt', prompt)
        found = self.wait_target(prompt, 30.0)
        print(f'prompt "{prompt}": ' + (f'found after {time.monotonic() - t0:.1f} s' if found
                                       else 'not found within 30 s'))
        self.show()
        return found

    def capture_reference(self):
        """Store the current features as the desired ones (the reference image)."""
        res = self.call('capture_reference')
        print(res.message)
        return res.success

    def go_start(self, start):
        """Move to one of the lab's start configurations (S1-S8) and detect again there."""
        prompt = self.text
        _, name = self.mover.go_to(self.target, start=start)
        self.start_name = name
        self.set_param('prompt', prompt)  # fresh detection in the new view
        found = self.wait_target(prompt, 40.0)
        print(f'{name}: target ' + ('found' if found else 'NOT found'))
        self.show()
        self.pose_error()
        return found

    def servo(self, timeout=25.0, lose_after=None, refresh=0.3):
        """Close the loop with a live plot; stops when converged, lost or after `timeout`.

        lose_after: move the target out of view after this many seconds (safe-stop test).
        """
        self.spin_for(0.3)
        self.set_param('max_rate', SERVO_RATE)
        self.rows, self.t0, self.converged, self.detections = [], time.monotonic(), False, 0
        self.call('start')
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.4))
        live = _Live(fig, extra=self.flow if _marimo() else None)
        outcome, conv, lost_t, mover = 'timeout', None, None, None
        moved = threading.Event()
        while time.monotonic() - self.t0 < timeout:
            self.spin_for(refresh)
            el = time.monotonic() - self.t0
            if lose_after is not None and mover is None and el >= lose_after:
                pose = self.mover.cfg['targets'][self.target]['pose']
                mover = threading.Thread(target=lambda: (move_model(
                    self.mover.cfg['targets'][self.target]['model'], 3.0, 3.0, 0.0), moved.set()),
                    daemon=True)
                mover.start()
            if lost_t is None and moved.is_set():
                lost_t = el
            self._draw(fig, axes, lost_t)
            live.update()
            if lose_after is not None:
                if lost_t is not None and el > lost_t + 5.0:
                    break
                continue
            if self.converged:
                outcome, conv = 'converged', el
                break
            if el > 2.0 and time.monotonic() - self.err_time > 3.5:  # controller gave up
                outcome = 'target lost'
                break
        self.spin_for(0.5)
        self.call('stop')
        self.set_param('max_rate', IDLE_RATE)
        self.last, self.rows = pd.DataFrame(self.rows, columns=SERIES_FIELDS), None
        self._draw(fig, axes, lost_t, self.last)
        live.update()
        if mover is not None:
            mover.join()
            x, y, z, yaw = pose
            move_model(self.mover.cfg['targets'][self.target]['model'], x, y, z, yaw)
            stop_t, drift = stop_metrics(self.last.values.tolist(), lost_t or el)
            print(f'target removed at {lost_t:.1f} s: arm still after {stop_t} s, '
                  f'drift {drift * 1000:.1f} mm (target put back)')
            return
        return self._summary(outcome, conv)

    def _draw(self, fig, axes, lost_t, df=None):
        df = pd.DataFrame(self.rows, columns=SERIES_FIELDS) if df is None else df
        for ax in axes:
            ax.clear()
            ax.grid(alpha=0.3)
            if lost_t is not None:
                ax.axvline(lost_t, color='tab:red', ls=':', lw=1)
        if df.empty:
            return
        for col in ['e_x', 'e_y', 'e_area', 'e_angle']:
            if df[col].notna().any():
                axes[0].plot(df.t, df[col], label=col)
        axes[0].set_title('image feature error')
        for col in ['v_x', 'v_y', 'v_z', 'w_z']:
            axes[1].plot(df.t, df[col], label=col)
        axes[1].set_title('camera velocity [m/s, rad/s]')
        axes[2].semilogy(df.t, df.gt_et * 1000, color='tab:purple')
        axes[2].axhline(POS_TOL * 1000, color='tab:red', ls='--', lw=1, label='5 mm tolerance')
        axes[2].set_title('position error, ground truth [mm]')
        for ax in axes:
            ax.set_xlabel('t [s]')
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8, loc='upper right')
        fig.tight_layout()

    def _summary(self, outcome, conv):
        df, g = self.last, self.gt or [math.nan] * 8
        rot = math.degrees(symmetric_rot_err(g[3], g[4], g[5], TARGETS[self.target][1]))
        last = df.iloc[-1]
        feat_err = math.sqrt(sum(last[c] ** 2 for c in ['e_x', 'e_y', 'e_area', 'e_angle']
                                 if not math.isnan(last[c])))
        ok = conv is not None and g[6] < POS_TOL and rot < ROT_TOL
        cam = df[['cam_x', 'cam_y', 'cam_z']].dropna().values
        path = float(np.linalg.norm(np.diff(cam, axis=0), axis=1).sum()) if len(cam) > 1 else 0.0
        v_max = float(np.linalg.norm(df[['v_x', 'v_y', 'v_z']].fillna(0).values, axis=1).max())
        row = {'target': self.target, 'start': self.start_name, 'prompt': self.text,
               'disturbance': self.disturbance, 'outcome': outcome,
               'time [s]': round(conv, 1) if conv else None,
               'pos error [mm]': round(g[6] * 1000, 2), 'rot error [deg]': round(rot, 2),
               'feature error': round(feat_err, 4), 'camera path [mm]': round(path * 1000),
               'peak speed [mm/s]': round(v_max * 1000), 'success': ok}
        self.history.append(row)
        print(f'{outcome}' + (f' after {conv:.1f} s' if conv else '') +
              f' | final position error {g[6] * 1000:.2f} mm, rotation {rot:.2f} deg | '
              f'feature error {feat_err:.4f} | camera path {path * 1000:.0f} mm | '
              f'peak speed {v_max * 1000:.0f} mm/s | '
              f'{"SUCCESS" if ok else "not within 5 mm / 2 deg"}')
        return row

    def trajectories(self):
        """Feature trajectory in the image and camera trajectory (top view) of the last run."""
        df = self.last
        fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
        ok = df.feat_valid == 1
        a.plot(df.u[ok], df.v[ok])
        a.scatter(df.u[ok].iloc[0], df.v[ok].iloc[0], color='tab:green', label='start', zorder=3)
        a.scatter(df.u[ok].iloc[-1], df.v[ok].iloc[-1], color='tab:red', label='end', zorder=3)
        a.set_xlim(0, 640), a.set_ylim(480, 0), a.set_aspect('equal'), a.legend()
        a.set_title('target centre in the image [px]')
        sc = b.scatter(df.cam_x * 1000, df.cam_y * 1000, c=df.cam_z * 1000, s=5)
        fig.colorbar(sc, ax=b, label='height z [mm]')
        b.set_aspect('equal'), b.set_xlabel('x [mm]'), b.set_ylabel('y [mm]')
        b.set_title('camera path, top view (world)')
        fig.tight_layout()
        _show_fig(fig)

    # --- disturbances ---------------------------------------------------------------------

    def occlude(self, fraction):
        """Grey plate just above the target covering `fraction` of its footprint."""
        spawn_occluder(self.mover.cfg['targets'][self.target], fraction)
        self.disturbance = f'occlusion {fraction:.0%}'
        print(f'occluder over {fraction:.0%} of the {self.target}')

    def light(self, factor):
        set_light(factor)
        self.disturbance = f'light {factor:.0%}'
        print(f'lights at {factor:.0%}')

    def clear(self):
        """Remove the occluder and restore the lights."""
        remove_occluder()
        set_light(1.0)
        self.disturbance = 'none'
        print('disturbances removed')

    def gazebo_gui(self, on):
        """Gazebo's own window as a client of the running headless server (the lab's world);
        closing it leaves the simulation running. Costs GPU while open."""
        proc = getattr(self, '_gui', None)
        if on and (proc is None or proc.poll() is not None):
            self._gui = subprocess.Popen(['gz', 'sim', '-g'], stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)
            print('Gazebo GUI opened (client of the running simulation)')
        elif not on and proc is not None and proc.poll() is None:
            proc.terminate()
            print('Gazebo GUI closed, the simulation keeps running headless')

    def set_detector(self, name, model='', timeout=900.0):
        """Swap the localisation model at runtime (perception frees the old one first).

        model: weights variant (e.g. IDEA-Research/grounding-dino-base) or, for name='vlm',
        the model served by Ollama (e.g. gemma4:e4b). The first load of a model downloads it.
        """
        from rcl_interfaces.msg import Parameter as RosParam, ParameterType, ParameterValue
        from rcl_interfaces.srv import GetParameters, SetParameters

        def string(v):
            return ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=v)
        want = f'{name}|{model}'
        req = SetParameters.Request(parameters=[RosParam(name='detector', value=string(name)),
                                                RosParam(name='detector_model',
                                                         value=string(model))])
        self.set_cli.wait_for_service(timeout_sec=30.0)
        fut = self.set_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        t0, state = time.monotonic(), ''
        while time.monotonic() - t0 < timeout:
            fut = self.get_cli.call_async(GetParameters.Request(names=['detector_loaded']))
            rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
            state = fut.result().values[0].string_value if fut.result() else ''
            if state == want or state.startswith('failed'):
                break
            self.spin_for(1.0)
        if state != want:
            print(f'switch to {name} {model} did not finish: {state or "timeout"}')
            return False
        self.params = self.perception_params()
        self.params['detector'] = model or name  # show the served model for VLMs
        print(f'model {model or name} ready after {time.monotonic() - t0:.0f} s')
        if self.text:
            self.set_param('prompt', self.text)  # fresh detection with the new model
        return True

    def set_verifier(self, name, timeout=300.0):
        """Switch the box verifier at runtime ('siglip' or 'none'); the first load downloads it.
        With it on, a prompt for an absent object ("purple bottle") gives no target at all."""
        from rcl_interfaces.msg import Parameter as RosParam, ParameterType, ParameterValue
        from rcl_interfaces.srv import GetParameters, SetParameters

        def loaded():
            fut = self.get_cli.call_async(GetParameters.Request(names=['verifier_loaded']))
            rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
            return fut.result().values[0].string_value if fut.result() else ''
        if loaded() == name:
            return True
        value = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=name)
        self.set_cli.wait_for_service(timeout_sec=30.0)
        fut = self.set_cli.call_async(
            SetParameters.Request(parameters=[RosParam(name='verifier', value=value)]))
        rclpy.spin_until_future_complete(self, fut, timeout_sec=10.0)
        t0, state = time.monotonic(), ''
        while time.monotonic() - t0 < timeout:
            state = loaded()
            if state == name or state.startswith('failed'):
                break
            self.spin_for(1.0)
        if state != name:
            print(f'verifier switch to {name} did not finish: {state or "timeout"}')
            return False
        print(f'verifier {name} active')
        if self.text:
            self.set_param('prompt', self.text)  # fresh detection, checked by the new verifier
        return True

    def set_latency(self, period=None, delay=None):
        """Detector period and an artificial extra delay per call (inference-time study)."""
        if period is not None:
            self.set_param('detector_period', period)
        if delay is not None:
            self.set_param('detector_delay', delay)
        p = self.perception_params()
        print(f'detector period {p["detector_period"]} s, extra delay {p["detector_delay"]} s')
