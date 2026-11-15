import numpy as np

from vs_perception.detectors.base import Detection, Detector, register_detector


@register_detector('sam3')
class Sam3(Detector):
    """SAM 3 via Hugging Face transformers (Meta, SAM licence): text prompt -> instance masks.

    The weights on Hugging Face are gated; the container needs HF_TOKEN of an account with access.
    """

    def load(self):
        import torch
        from transformers import Sam3Model, Sam3Processor

        model_id = self.options.get('model_id', 'facebook/sam3')
        self._torch = torch
        self._processor = Sam3Processor.from_pretrained(model_id)
        self._model = Sam3Model.from_pretrained(model_id).to(self.device).eval()

    def detect(self, image_rgb, prompt):
        inputs = self._processor(images=image_rgb, text=prompt, return_tensors='pt').to(self.device)
        with self._torch.inference_mode():
            outputs = self._model(**inputs)
        result = self._processor.post_process_instance_segmentation(
            outputs, threshold=self.score_threshold, mask_threshold=0.5,
            target_sizes=inputs.get('original_sizes').tolist())[0]
        return [Detection(tuple(np.asarray(b.cpu(), dtype=float)), float(s), prompt,
                          mask=np.asarray(m.cpu(), dtype=bool))
                for b, s, m in zip(result['boxes'], result['scores'], result['masks'])]
