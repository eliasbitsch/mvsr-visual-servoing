from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from vs_perception.registry import Registry

DETECTORS = Registry('detector')
register_detector = DETECTORS.register


@dataclass
class Detection:
    box: tuple  # x_min, y_min, x_max, y_max in pixels
    score: float
    label: str = ''
    mask: Optional[np.ndarray] = None  # bool HxW, if the model produces one


class Detector(ABC):
    """Open-vocabulary localisation: image + text prompt -> boxes.

    Heavy imports (torch, transformers, ...) belong in load(), so listing or
    creating detectors works without a GPU stack.
    """

    def __init__(self, device='cuda', score_threshold=0.3, **options):
        self.device = device
        self.score_threshold = score_threshold
        self.options = options

    @abstractmethod
    def load(self):
        """Download / load weights. Called once, before the first detect()."""

    @abstractmethod
    def detect(self, image_rgb: np.ndarray, prompt: str) -> list:
        """Return Detection objects above score_threshold, any order."""
