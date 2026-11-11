"""Fast frame-to-frame trackers that bridge the gap between two detector calls."""
from abc import ABC, abstractmethod

import threading

import cv2
import numpy as np

from vs_perception import sam2_utils
from vs_perception.registry import Registry, weights_path

TRACKERS = Registry('tracker')


class Tracker(ABC):
    # Per-frame target mask if the tracker produces one (used for features instead of the box)
    mask = None
    # True if one update costs more than a camera period: the node then feeds only the newest frame
    slow = False

    def load(self):
        """Load heavy resources once, before the first frame."""

    @abstractmethod
    def init(self, image_rgb, box):
        """Start tracking box (x_min, y_min, x_max, y_max) in image_rgb."""

    @abstractmethod
    def update(self, image_rgb):
        """Return the new box, or None if the target is lost."""


class _OpenCvTracker(Tracker):
    factory = None

    def __init__(self, **_):
        self._tracker = None

    def init(self, image_rgb, box):
        x0, y0, x1, y1 = box
        self._tracker = self.factory()
        self._tracker.init(image_rgb, (int(x0), int(y0), int(x1 - x0), int(y1 - y0)))

    def update(self, image_rgb):
        if self._tracker is None:
            return None
        ok, (x, y, w, h) = self._tracker.update(image_rgb)
        if not ok or w <= 1 or h <= 1:
            return None
        return (float(x), float(y), float(x + w), float(y + h))


@TRACKERS.register('csrt')
class CsrtTracker(_OpenCvTracker):
    """Accurate, ~30-60 fps at 640x480 on a laptop CPU."""
    factory = staticmethod(cv2.TrackerCSRT.create)


@TRACKERS.register('kcf')
class KcfTracker(_OpenCvTracker):
    """Faster than CSRT, does not adapt the box size (bad for approach motions)."""
    factory = staticmethod(cv2.TrackerKCF.create)


@TRACKERS.register('none')
class NoTracker(Tracker):
    """Baseline for the latency study: features only exist on detector frames."""

    def __init__(self, **_):
        pass

    def init(self, image_rgb, box):
        pass

    def update(self, image_rgb):
        return None


@TRACKERS.register('sam2')
class Sam2Tracker(Tracker):
    """SAM 2.1 tiny with a memory bank (Ultralytics, Apache 2.0 weights).

    Initialised with a detector box, then segments every frame by attending to the prompted
    frame and to the last confident masks. Gives a mask (centroid, area, angle) per update.
    One predictor and model are shared by all instances (the node creates a new tracker for
    every detection); each instance only owns its memory bank, swapped in under a lock.
    """
    slow = True
    _lock = threading.Lock()
    _predictor = None

    def __init__(self, device='cuda', weights='sam2.1_t.pt', min_score=0.05,
                 refresh_interval=15, max_memory=4, **_):
        self.device = device
        self.weights = weights
        self.min_score = min_score
        self.refresh_interval = refresh_interval
        self.max_memory = max_memory
        self._bank = []
        self._ref_area = 0.0
        self._since_refresh = 0
        self.score = 0.0

    def load(self):
        with self._lock:
            self._get_predictor()
        # Warm-up: the first calls load the weights and initialise CUDA
        blank = np.zeros((480, 640, 3), np.uint8)
        self.init(blank, (200, 150, 300, 250))
        self.update(blank)

    def _get_predictor(self):
        cls = Sam2Tracker
        if cls._predictor is None:
            from ultralytics.models.sam.predict import SAM2DynamicInteractivePredictor
            cls._predictor = SAM2DynamicInteractivePredictor(
                overrides=dict(task='segment', mode='predict', imgsz=1024, save=False,
                               verbose=False, conf=0.0, device=self.device,
                               model=weights_path(self.weights)),
                max_obj_num=1)
        return cls._predictor

    def _run(self, image_rgb, **prompts):
        """One predictor call with this instance's memory bank; returns (mask, score)."""
        predictor = self._get_predictor()
        predictor.memory_bank = self._bank
        # Ultralytics expects BGR numpy input
        result = predictor(source=np.ascontiguousarray(image_rgb[:, :, ::-1]), **prompts)[0]
        if result.masks is None or len(result.masks.data) == 0:
            return None, 0.0
        mask = result.masks.data[0].cpu().numpy() > 0.5
        score = float(result.boxes.conf[0]) if result.boxes is not None and len(result.boxes) else 0.0
        return mask, score

    def init(self, image_rgb, box):
        with self._lock:
            predictor = self._get_predictor()
            self._bank = []
            predictor.obj_idx_set = set()
            mask, self.score = self._run(image_rgb, bboxes=[[float(v) for v in box]],
                                         obj_ids=[0], update_memory=True)
        self.mask = mask
        self._ref_area = float(np.count_nonzero(mask)) if mask is not None else 0.0
        self._since_refresh = 0

    def update(self, image_rgb):
        if not self._bank:
            return None
        with self._lock:
            self._get_predictor().obj_idx_set = {0}
            mask, self.score = self._run(image_rgb)
            if not sam2_utils.mask_is_valid(mask, self.score, self._ref_area, self.min_score):
                self.mask = None
                return None
            self._since_refresh += 1
            if sam2_utils.refresh_due(self._since_refresh, self.refresh_interval, self.score,
                                      3 * self.min_score):
                # Mask prompts break inside Ultralytics' update_memory, so prompt with the box
                box = sam2_utils.mask_to_box(mask)
                self._run(image_rgb, bboxes=[list(box)], obj_ids=[0], update_memory=True)
                sam2_utils.prune_memory(self._bank, self.max_memory)
                self._since_refresh = 0
        self.mask = mask
        return sam2_utils.mask_to_box(mask)
