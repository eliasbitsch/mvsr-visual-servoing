import numpy as np

from vs_perception.detectors.base import Detection, Detector, register_detector


@register_detector('yoloe')
class YoloE(Detector):
    """YOLOE via Ultralytics (AGPL-3.0): real-time open-vocabulary detection with masks."""

    # Its scores on the simulated objects are very low (blue box 0.085, red can 0.047 with the
    # small model, measured 17.11.), so even YOLO-World's 0.1 found nothing
    default_threshold = 0.03

    def load(self):
        import os

        from ultralytics import YOLOE, settings

        from vs_perception.registry import weights_path
        if os.environ.get('VS_WEIGHTS_DIR'):  # the text encoder lands next to the weights
            settings.update({'weights_dir': os.environ['VS_WEIGHTS_DIR']})
        self.score_threshold = min(self.score_threshold, self.default_threshold)
        self._model = YOLOE(weights_path(self.options.get('model_id', 'yoloe-11s-seg.pt')))
        self._prompt = None

    def detect(self, image_rgb, prompt):
        if prompt != self._prompt:
            # Text embedding once per prompt, as for YOLO-World
            self._model.set_classes([prompt], self._model.get_text_pe([prompt]))
            self._prompt = prompt
        result = self._model.predict(np.ascontiguousarray(image_rgb[:, :, ::-1]),
                                     conf=self.score_threshold, device=self.device,
                                     verbose=False)[0]
        boxes, masks = result.boxes, result.masks
        h, w = image_rgb.shape[:2]
        out = []
        for i, (b, s) in enumerate(zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy())):
            mask = None
            if masks is not None:
                m = masks.data[i].cpu().numpy()
                if m.shape != (h, w):  # masks come at the network resolution
                    import cv2
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                mask = m > 0.5
            out.append(Detection(tuple(map(float, b)), float(s), prompt, mask=mask))
        return out
