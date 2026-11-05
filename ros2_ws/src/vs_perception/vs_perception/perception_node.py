"""Target localisation: slow foundation model in a worker thread, fast tracker on every frame.

The detector result belongs to the frame it was computed on, which is already old when it
arrives. The tracker is therefore re-initialised on that buffered frame and replayed through
all frames since, so the latency of the model does not show up as a jump in the features.
"""
import queue
import threading
import time
from collections import deque

import cv2
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image
from vs_interfaces.msg import ObjectFeatures

from vs_perception import features
from vs_perception.detectors import DETECTORS
from vs_perception.image_utils import from_rgb, to_rgb
from vs_perception.refiners import REFINERS
from vs_perception.trackers import TRACKERS, NoTracker


class PerceptionNode(Node):

    def __init__(self):
        super().__init__('perception_node')
        params = {
            'detector': 'grounding_dino', 'prompt': 'red box', 'detector_period': 1.0,
            'score_threshold': 0.3, 'device': 'cuda', 'tracker': 'csrt', 'refiner': 'none',
            'image_topic': '/camera/camera/color/image_raw', 'max_missed_detections': 3,
            'frame_buffer_size': 150, 'publish_debug_image': True,
        }
        for name, default in params.items():
            self.declare_parameter(name, default)
        p = {name: self.get_parameter(name).value for name in params}
        self.prompt = p['prompt']
        self.period = p['detector_period']
        self.max_missed = p['max_missed_detections']
        self.debug = p['publish_debug_image']

        self.get_logger().info(f'loading detector "{p["detector"]}" on {p["device"]} ...')
        t0 = time.monotonic()
        self.detector = DETECTORS.create(p['detector'], device=p['device'],
                                         score_threshold=p['score_threshold'])
        self.detector.load()
        self.refiner = REFINERS.create(p['refiner'], device=p['device'])
        self.refiner.load()
        self.tracker = TRACKERS.create(p['tracker'])
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
        self.create_subscription(Image, p['image_topic'], self.on_image, qos_profile_sensor_data)
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
            if frame is None or time.monotonic() < next_run:
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
            self.results.put((seq, stamp, prompt, dets, time.monotonic() - t0))

    def apply_detections(self, current_seq):
        """Returns True if a fresh detection was applied in this call."""
        fresh = False
        while not self.results.empty():
            seq, stamp, prompt, dets, latency = self.results.get_nowait()
            if prompt != self.prompt:
                continue
            self.latency = latency
            if not dets:
                self.missed += 1
                if self.missed >= self.max_missed:
                    self.track_box = None
                continue
            self.missed = 0
            best = max(dets, key=lambda d: d.score)
            self.score = best.score
            self.det_stamp = stamp
            self.det_mask = best.mask
            self.reinit_tracker(seq, best.box, current_seq)
            fresh = True
        return fresh

    def reinit_tracker(self, det_seq, box, current_seq):
        buffered = [f for f in self.frames if f[0] >= det_seq]
        if not buffered or buffered[0][0] != det_seq:
            # Detection frame already dropped from the buffer: accept the small error
            buffered = [self.frames[-1]]
        self.tracker.init(buffered[0][2], box)
        self.track_box, self.track_seq = box, buffered[0][0]
        if isinstance(self.tracker, NoTracker):
            return
        for seq, _, img in buffered[1:]:
            if seq > current_seq:
                break
            box = self.tracker.update(img)
            if box is None:
                self.track_box = None
                return
            self.track_box, self.track_seq = box, seq

    def on_image(self, msg):
        img = to_rgb(msg)
        self.seq += 1
        self.frames.append((self.seq, msg.header.stamp, img))
        with self.latest_lock:
            self.latest = (self.seq, msg.header.stamp, img)

        fresh = self.apply_detections(self.seq)
        tracking = not isinstance(self.tracker, NoTracker)
        if self.track_box is not None and tracking and self.track_seq < self.seq:
            self.track_box = self.tracker.update(img)
            self.track_seq = self.seq

        out = ObjectFeatures(header=msg.header, prompt=self.prompt,
                             source='detector' if fresh else 'tracker')
        box = self.track_box
        # Without a tracker, features only exist at the moment a detection arrives
        if box is not None and (tracking or fresh):
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
        text = (f'{self.detector.registry_name} "{self.prompt}" {f.source} '
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
