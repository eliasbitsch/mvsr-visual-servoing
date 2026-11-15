import numpy as np

from vs_perception.detectors.base import Detection, Detector, register_detector


@register_detector('yolo_world')
class YoloWorld(Detector):
    """YOLO-World via Ultralytics (AGPL-3.0). Real-time, fits on 4 GB GPUs."""

    # Its scores on the simulated objects stay far below the shared 0.3 (red can ~0.02,
    # blue box ~0.2); 0.1 is the usual setting for open-vocabulary YOLO
    default_threshold = 0.1

    def load(self):
        import os

        from ultralytics import YOLOWorld, settings

        from vs_perception.registry import weights_path
        # CLIP (338 MB) goes to <weights_dir>/clip; the default is relative to the working
        # directory, so every container downloaded it again
        if os.environ.get('VS_WEIGHTS_DIR'):
            settings.update({'weights_dir': os.environ['VS_WEIGHTS_DIR']})
        self.score_threshold = min(self.score_threshold, self.default_threshold)
        self._model = YOLOWorld(weights_path(self.options.get('weights', 'yolov8s-worldv2.pt')))
        self._prompt = None

    def detect(self, image_rgb, prompt):
        if prompt != self._prompt:
            # Re-encodes the text with CLIP, so only when the prompt changes
            self._model.set_classes([prompt])
            self._prompt = prompt
        result = self._model.predict(np.ascontiguousarray(image_rgb[:, :, ::-1]),
                                     conf=self.score_threshold, device=self.device,
                                     verbose=False)[0]
        boxes = result.boxes
        return [Detection(tuple(map(float, b)), float(s), prompt)
                for b, s in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy())]
