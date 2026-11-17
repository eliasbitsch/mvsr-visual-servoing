"""Pure numpy helpers of the SAM 2 tracker, kept free of torch so they can be unit tested."""
import numpy as np


def mask_to_box(mask, min_area=16):
    """Tight (x_min, y_min, x_max, y_max) around a bool mask, None if it is (nearly) empty."""
    if mask is None:
        return None
    ys, xs = np.nonzero(mask)
    if xs.size < min_area:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def mask_is_valid(mask, score, ref_area, min_score=0.05, min_area_ratio=0.2,
                  max_area_ratio=4.0):
    """False when SAM 2 reports the object as absent or the mask collapsed / leaked.

    ref_area is the mask area at initialisation. An approach motion enlarges the object, hence
    the generous upper bound; it only catches a mask that flooded the background.
    """
    if mask is None or score < min_score:
        return False
    area = float(np.count_nonzero(mask))
    if area < 16 or ref_area <= 0:
        return False
    return min_area_ratio <= area / ref_area <= max_area_ratio


def refresh_due(updates_since_refresh, interval, score, min_score):
    """Memory refresh from a predicted mask: periodic, and only while SAM 2 is confident."""
    return interval > 0 and updates_since_refresh >= interval and score >= min_score


def prune_memory(bank, max_len):
    """Keep the prompted first entry plus the newest ones, in place."""
    if max_len >= 1 and len(bank) > max_len:
        del bank[1:len(bank) - max_len + 1]
    return bank
