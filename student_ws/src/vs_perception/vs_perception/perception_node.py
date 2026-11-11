"""Target localisation: slow foundation model in a worker thread, fast tracker on every frame.

The detector result belongs to the frame it was computed on, which is already old when it
arrives. The worker therefore initialises a fresh tracker on that buffered frame and replays it
through all frames since; only then is it swapped in. The running tracker keeps publishing in
the meantime, so neither the model latency nor the replay shows up as a gap or jump.
"""
import queue
import threading
import time
from collections import deque

import cv2
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image
from vs_interfaces.msg import ObjectFeatures

from vs_perception import features
from vs_perception.detectors import DETECTORS
from vs_perception.image_utils import from_rgb, to_rgb
from vs_perception.refiners import REFINERS
from vs_perception.trackers import TRACKERS, NoTracker


cv2.setNumThreads(2)  # default pool = all cores, mostly spinning; CSRT does not need more


class PerceptionNode(Node):

    def __init__(self):
        super().__init__('perception_node')
        params = {
            'detector': 'grounding_dino', 'prompt': 'red can', 'detector_period': 1.0,
            'score_threshold': 0.3, 'device': 'cuda', 'tracker': 'csrt', 'refiner': 'none',
            'image_topic': '/camera/image_raw', 'max_missed_detections': 3,
            'frame_buffer_size': 150, 'publish_debug_image': True,
            'detect_on_loss': False,
        }
        for name, default in params.items():
            self.declare_parameter(name, default)
        p = {name: self.get_parameter(name).value for name in params}
        self.prompt = p['prompt']
        self.period = p['detector_period']
        self.max_missed = p['max_missed_detections']
        self.debug = p['publish_debug_image']
        self.detect_on_loss = p['detect_on_loss']
        self.device = p['device']

        self.get_logger().info(f'loading detector "{p["detector"]}" on {p["device"]} ...')
        t0 = time.monotonic()
        self.detector = DETECTORS.create(p['detector'], device=p['device'],
                                         score_threshold=p['score_threshold'])
        self.detector.load()
        self.refiner = REFINERS.create(p['refiner'], device=p['device'])
        self.refiner.load()
        self.tracker_name = p['tracker']
        self.tracker = TRACKERS.create(self.tracker_name, device=self.device)
        self.tracker.load()
        self.tracking = not isinstance(self.tracker, NoTracker)
        self.get_logger().info(f'models ready after {time.monotonic() - t0:.1f} s')

        self.frames = deque(maxlen=p['frame_buffer_size'])
        self.seq = 0
        self.latest = None
        self.latest_lock = threading.Lock()
        self.results = queue.Queue()

        self.track_box = None
        self.track_seq = -1
        self.det_stamp = None
        self.det_mask = None
        self.score = 0.0
        self.latency = 0.0
        self.missed = 0

        self.pub = self.create_publisher(ObjectFeatures, '/vs/features', 10)
        self.debug_pub = self.create_publisher(Image, '/vs/debug_image', 2)
        # A slow tracker would otherwise work through a backlog of stale frames
        qos = qos_profile_sensor_data
        if self.tracker.slow:
            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, p['image_topic'], self.on_image, qos)
        self.add_on_set_parameters_callback(self.on_params)

        self._stop = threading.Event()
        threading.Thread(target=self.detector_loop, daemon=True).start()

    def on_params(self, changes):
        for c in changes:
            if c.name == 'prompt' and c.value != self.prompt:
                self.prompt = c.value
                self.track_box = None
                self.get_logger().info(f'new prompt "{self.prompt}"')
            elif c.name == 'detector_period':
                self.period = c.value
        return SetParametersResult(successful=True)

    def detector_loop(self):
        next_run = 0.0
        while not self._stop.is_set():
            with self.latest_lock:
                frame = self.latest
            # detect_on_loss: the model only runs while there is no track to follow
            idle = self.detect_on_loss and self.track_box is not None
            if frame is None or idle or time.monotonic() < next_run:
                time.sleep(0.005)
                continue
            next_run = time.monotonic() + self.period
            seq, stamp, img = frame
            prompt = self.prompt
            t0 = time.monotonic()
            try:
                dets = self.detector.detect(img, prompt)
            except Exception as exc:  # keep the node alive, the tracker may still run
                self.get_logger().error(f'detector failed: {exc}')
                dets = []
            latency = time.monotonic() - t0
            best = max(dets, key=lambda d: d.score) if dets else None
            replay = self.replay_tracker(seq, best.box) if best else None
            self.results.put((stamp, prompt, best, latency, replay))

    def buffered_since(self, seq):
        with self.latest_lock:
            return [f for f in self.frames if f[0] >= seq]

    def replay_tracker(self, det_seq, box):
        """Fresh tracker started on the detection frame and run up to the newest frame.
        Returns (tracker, box or None if lost, seq of the last frame it saw)."""
        tracker = TRACKERS.create(self.tracker_name, device=self.device)
        buffered = self.buffered_since(det_seq)
        if not buffered or buffered[0][0] != det_seq:
            # Detection frame already dropped from the buffer: accept the small error
            buffered = self.buffered_since(self.seq)
        tracker.init(buffered[0][2], box)
        last = buffered[0][0]
        if not self.tracking:
            return tracker, box, last
        rest = buffered[1:]
        if tracker.slow:  # skip the frames in between, the newest one is enough
            rest = rest[-1:]
        for seq, _, img in rest:
            box = tracker.update(img)
            if box is None:
                return tracker, None, seq
            last = seq
        return tracker, box, last

    def apply_detections(self):
        """Returns True if a fresh detection was applied in this call."""
        fresh = False
        while not self.results.empty():
            stamp, prompt, best, latency, replay = self.results.get_nowait()
            if prompt != self.prompt:
                continue
            self.latency = latency
            if best is None:
                self.missed += 1
                if self.missed >= self.max_missed:
                    self.track_box = None
                continue
            self.missed = 0
            self.score = best.score
            self.det_stamp = stamp
            self.det_mask = best.mask
            self.tracker, self.track_box, self.track_seq = replay
            fresh = True
        return fresh

    def on_image(self, msg):
        img = to_rgb(msg)
        with self.latest_lock:
            self.seq += 1
            self.frames.append((self.seq, msg.header.stamp, img))
            self.latest = (self.seq, msg.header.stamp, img)

        fresh = self.apply_detections()
        tracking = self.tracking
        # Catch up the frames that arrived while the worker replayed (usually one or two)
        if self.track_box is not None and tracking and self.track_seq < self.seq:
            pending = self.buffered_since(self.track_seq + 1)
            for seq, _, frame in (pending[-1:] if self.tracker.slow else pending):
                self.track_box = self.tracker.update(frame)
                self.track_seq = seq
                if self.track_box is None:
                    break

        out = ObjectFeatures(header=msg.header, prompt=self.prompt,
                             source='detector' if fresh else 'tracker')
        box = self.track_box
        # Without a tracker, features only exist at the moment a detection arrives
        if box is not None and (tracking or fresh):
            mask = self.tracker.mask if tracking else None
            if mask is None:
                mask = self.det_mask if fresh and self.det_mask is not None else \
                    self.refiner.refine(img, box)
            f = features.from_mask(mask) if mask is not None else None
            f = f or features.from_box(box)
            out.valid = True
            out.u, out.v, out.area, out.angle = f.u, f.v, f.area, f.angle
            out.bbox = [float(x) for x in box]
            out.score = self.score
        if self.det_stamp is not None:
            out.detection_age = (Time.from_msg(msg.header.stamp)
                                 - Time.from_msg(self.det_stamp)).nanoseconds * 1e-9
        out.detector_latency = self.latency
        self.pub.publish(out)

        if self.debug and self.debug_pub.get_subscription_count() > 0:
            self.debug_pub.publish(from_rgb(self.draw(img, out), msg.header))

    def draw(self, img, f):
        vis = img.copy()
        if f.valid:
            x0, y0, x1, y1 = map(int, f.bbox)
            color = (0, 255, 0) if f.source == 'detector' else (255, 200, 0)
            cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
            cv2.circle(vis, (int(f.u), int(f.v)), 4, color, -1)
        text = (f'{self.detector.registry_name}+{self.tracker_name} "{self.prompt}" {f.source} '
                f'lat {f.detector_latency * 1000:.0f} ms age {f.detection_age:.2f} s')
        cv2.putText(vis, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return vis

    def destroy_node(self):
        self._stop.set()
        super().destroy_node()


def main():
    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
