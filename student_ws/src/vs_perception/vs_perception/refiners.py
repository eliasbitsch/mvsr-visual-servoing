"""Turn a box into a mask, so orientation and a cleaner centroid can be computed."""
from abc import ABC, abstractmethod

import cv2
import numpy as np

from vs_perception.registry import Registry

REFINERS = Registry('refiner')


class MaskRefiner(ABC):

    def __init__(self, device='cuda', **options):
        self.device = device
        self.options = options

    def load(self):
        pass

    @abstractmethod
    def refine(self, image_rgb, box):
        """Return a bool mask HxW for the object inside box, or None."""


@REFINERS.register('none')
class NoRefiner(MaskRefiner):

    def refine(self, image_rgb, box):
        return None


@REFINERS.register('sam2')
class Sam2Refiner(MaskRefiner):
    """SAM 2.1 tiny with a box prompt, via Ultralytics (Meta, Apache 2.0 weights)."""

    def load(self):
        from ultralytics import SAM

        from vs_perception.registry import weights_path
        self._model = SAM(weights_path(self.options.get('weights', 'sam2.1_t.pt')))

    def refine(self, image_rgb, box):
        result = self._model(np.ascontiguousarray(image_rgb[:, :, ::-1]), bboxes=[list(box)],
                             device=self.device, verbose=False)[0]
        if result.masks is None or len(result.masks.data) == 0:
            return None
        mask = result.masks.data[0].cpu().numpy()
        h, w = image_rgb.shape[:2]
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return mask.astype(bool)
