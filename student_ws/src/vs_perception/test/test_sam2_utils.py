import numpy as np

from vs_perception import sam2_utils as su
from vs_perception.trackers import TRACKERS, NoTracker, Sam2Tracker


def rect_mask(x0=20, y0=10, x1=60, y1=40, shape=(100, 120)):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


def test_mask_to_box_is_tight_and_exclusive_at_max():
    assert su.mask_to_box(rect_mask()) == (20.0, 10.0, 60.0, 40.0)


def test_mask_to_box_empty_or_tiny():
    assert su.mask_to_box(None) is None
    assert su.mask_to_box(np.zeros((50, 50), bool)) is None
    tiny = np.zeros((50, 50), bool)
    tiny[5:7, 5:7] = True
    assert su.mask_to_box(tiny) is None


def test_mask_valid_accepts_normal_and_growing_target():
    ref = float(rect_mask().sum())
    assert su.mask_is_valid(rect_mask(), 0.3, ref)
    assert su.mask_is_valid(rect_mask(10, 5, 80, 60), 0.3, ref)  # approach: ~3x larger


def test_mask_valid_rejects_low_score_collapse_and_leak():
    ref = float(rect_mask().sum())
    assert not su.mask_is_valid(rect_mask(), 0.01, ref)
    assert not su.mask_is_valid(None, 0.9, ref)
    assert not su.mask_is_valid(rect_mask(20, 10, 30, 14), 0.9, ref)  # collapsed
    assert not su.mask_is_valid(np.ones((100, 120), bool), 0.9, ref)  # flooded
    assert not su.mask_is_valid(rect_mask(), 0.9, 0.0)


def test_refresh_due():
    assert not su.refresh_due(5, 15, 0.5, 0.1)
    assert su.refresh_due(15, 15, 0.5, 0.1)
    assert not su.refresh_due(15, 15, 0.05, 0.1)  # not confident enough
    assert not su.refresh_due(100, 0, 0.9, 0.1)  # disabled


def test_prune_memory_keeps_prompt_and_newest():
    bank = list(range(7))
    su.prune_memory(bank, 4)
    assert bank == [0, 4, 5, 6]
    short = [0, 1]
    su.prune_memory(short, 4)
    assert short == [0, 1]


def test_registry_has_sam2_and_old_trackers():
    assert {'csrt', 'kcf', 'none', 'sam2'} <= set(TRACKERS.names())
    assert isinstance(TRACKERS.create('none', device='cpu'), NoTracker)


def test_sam2_tracker_construct_does_not_touch_gpu_and_update_before_init_is_none():
    t = TRACKERS.create('sam2', device='cpu')
    assert isinstance(t, Sam2Tracker)
    assert t.slow and t.mask is None
    assert t.update(np.zeros((48, 64, 3), np.uint8)) is None
