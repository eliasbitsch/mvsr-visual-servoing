"""Image features for IBVS: centroid, area and principal-axis angle of the target."""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Features:
    u: float
    v: float
    area: float
    angle: float


def from_box(box):
    x0, y0, x1, y1 = box
    return Features(u=(x0 + x1) / 2, v=(y0 + y1) / 2, area=max(x1 - x0, 0) * max(y1 - y0, 0),
                    angle=0.0)


def from_mask(mask):
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    if m['m00'] < 1:
        return None
    # Principal axis from the central second-order moments. Ill-defined for
    # rotationally symmetric shapes (mu20 ~ mu02, mu11 ~ 0), e.g. a cube seen from above.
    angle = 0.5 * np.arctan2(2 * m['mu11'], m['mu20'] - m['mu02'])
    return Features(u=m['m10'] / m['m00'], v=m['m01'] / m['m00'], area=m['m00'], angle=angle)
