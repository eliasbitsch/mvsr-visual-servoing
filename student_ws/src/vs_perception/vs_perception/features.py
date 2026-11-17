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
    # False when the shape has no defined principal axis (box only, or a round/square mask)
    angle_valid: bool = False

# Ratio of the principal second moments below which the axis is noise: a circle or square has
# 1.0, the blue box (8 x 6 cm) about 1.8. Above this, the angle is used for yaw control.
MIN_ELONGATION = 1.2


def from_box(box):
    x0, y0, x1, y1 = box
    return Features(u=(x0 + x1) / 2, v=(y0 + y1) / 2, area=max(x1 - x0, 0) * max(y1 - y0, 0),
                    angle=0.0, angle_valid=False)


def from_mask(mask):
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    if m['m00'] < 1:
        return None
    # Principal axis from the central second-order moments. Ill-defined for rotationally
    # symmetric shapes (mu20 ~ mu02, mu11 ~ 0): a can or ball from above has no yaw to servo.
    mu20, mu02, mu11 = m['mu20'], m['mu02'], m['mu11']
    angle = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    root = np.hypot(mu20 - mu02, 2 * mu11)
    elongation = (mu20 + mu02 + root) / max(mu20 + mu02 - root, 1e-9)
    return Features(u=m['m10'] / m['m00'], v=m['m01'] / m['m00'], area=m['m00'], angle=angle,
                    angle_valid=bool(elongation > MIN_ELONGATION))
