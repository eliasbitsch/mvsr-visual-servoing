import numpy as np

from vs_perception.detectors.base import Detection, Detector, register_detector


@register_detector('grounding_dino')
class GroundingDino(Detector):
    """Grounding DINO via Hugging Face transformers (IDEA-Research, Apache 2.0)."""

    def load(self):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        model_id = self.options.get('model_id', 'IDEA-Research/grounding-dino-tiny')
        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self._model.to(self.device).eval()

    def detect(self, image_rgb, prompt):
        # The model expects lower-case phrases terminated by a dot
        text = prompt.lower().strip().rstrip('.') + '.'
        inputs = self._processor(images=image_rgb, text=text, return_tensors='pt').to(self.device)
        with self._torch.inference_mode():
            outputs = self._model(**inputs)
        h, w = image_rgb.shape[:2]
        result = self._processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=self.score_threshold,
            text_threshold=self.options.get('text_threshold', 0.25), target_sizes=[(h, w)])[0]
        labels = result.get('text_labels', result.get('labels'))
        return [Detection(tuple(np.asarray(b.cpu(), dtype=float)), float(s), str(lbl))
                for b, s, lbl in zip(result['boxes'], result['scores'], labels)]
