import numpy as np

from vs_perception.detectors.base import Detection
from vs_perception.verifiers import VERIFIERS, Verifier, crop_with_context


class FixedVerifier(Verifier):
    """Scores from a lookup on the box's x_min, no model."""
    threshold = 0.5

    def scores(self, image_rgb, prompt, boxes):
        return [self.options['by_x'][b[0]] for b in boxes]


def test_crop_is_square_with_context_and_clipped():
    img = np.zeros((480, 640, 3), np.uint8)
    crop = crop_with_context(img, (100, 200, 140, 220), context=0.25)
    assert crop.shape[0] == crop.shape[1] == 60      # longer side 40 px * 1.5
    edge = crop_with_context(img, (0, 0, 40, 40))
    assert edge.shape[:2] == (46, 46)                # clipped at the image corner, not shifted
    tiny = crop_with_context(img, (300, 300, 302, 302), context=0.0)
    assert tiny.shape[:2] == (32, 32)                # min_side keeps the crop usable


def test_filter_keeps_boxes_at_or_above_threshold_and_records_scores():
    v = FixedVerifier(by_x={10: 0.9, 20: 0.5, 30: 0.1})
    dets = [Detection((x, 0, x + 5, 5), 1.0) for x in (10, 20, 30)]
    kept = v.filter(np.zeros((50, 50, 3), np.uint8), 'red can', dets)
    assert [d.box[0] for d in kept] == [10, 20]
    assert [d.verify_score for d in dets] == [0.9, 0.5, 0.1]


def test_threshold_argument_overrides_the_default():
    v = FixedVerifier(threshold=0.95, by_x={10: 0.9})
    assert v.filter(np.zeros((50, 50, 3), np.uint8), 'x', [Detection((10, 0, 15, 5), 1.0)]) == []
    assert FixedVerifier(by_x={}).threshold == 0.5


def test_none_keeps_everything_and_empty_input_is_fine():
    v = VERIFIERS.create('none')
    dets = [Detection((1, 1, 5, 5), 0.3)]
    assert v.filter(np.zeros((10, 10, 3), np.uint8), 'x', dets) == dets
    assert v.filter(np.zeros((10, 10, 3), np.uint8), 'x', []) == []
    assert 'siglip' in VERIFIERS.names()
