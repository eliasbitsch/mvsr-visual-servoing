"""Fast frame-to-frame trackers that bridge the gap between two detector calls."""
from abc import ABC, abstractmethod

import cv2

from vs_perception.registry import Registry

TRACKERS = Registry('tracker')


class Tracker(ABC):

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
