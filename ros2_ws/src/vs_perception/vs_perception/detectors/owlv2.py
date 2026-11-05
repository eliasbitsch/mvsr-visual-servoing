import numpy as np

from vs_perception.detectors.base import Detection, Detector, register_detector


@register_detector('owlv2')
class OwlV2(Detector):
    """OWLv2 via Hugging Face transformers (Google, Apache 2.0)."""

    def load(self):
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        model_id = self.options.get('model_id', 'google/owlv2-base-patch16-ensemble')
        self._torch = torch
        self._processor = Owlv2Processor.from_pretrained(model_id)
        self._model = Owlv2ForObjectDetection.from_pretrained(model_id).to(self.device).eval()

    def detect(self, image_rgb, prompt):
        inputs = self._processor(text=[[prompt]], images=image_rgb,
                                 return_tensors='pt').to(self.device)
        with self._torch.inference_mode():
            outputs = self._model(**inputs)
        # OWLv2 pads the image to a square, so boxes are relative to the padded size
        side = max(image_rgb.shape[:2])
        result = self._processor.post_process_object_detection(
            outputs, threshold=self.score_threshold, target_sizes=[(side, side)])[0]
        return [Detection(tuple(np.asarray(b.cpu(), dtype=float)), float(s), prompt)
                for b, s in zip(result['boxes'], result['scores'])]
