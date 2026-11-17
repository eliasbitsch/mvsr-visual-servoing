import numpy as np

from vs_perception.detectors.base import Detection, Detector, register_detector

TASK = '<CAPTION_TO_PHRASE_GROUNDING>'


@register_detector('florence2')
class Florence2(Detector):
    """Florence-2 via Hugging Face transformers (Microsoft, MIT), phrase grounding.

    Small enough for the CPU. Grounding gives no confidence, every box gets score 1.0.
    """

    def load(self):
        import torch
        from transformers import AutoProcessor, Florence2ForConditionalGeneration

        model_id = self.options.get('model_id', 'florence-community/Florence-2-base')
        self._torch = torch
        self._dtype = torch.float16 if self.device.startswith('cuda') else torch.float32
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = Florence2ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=self._dtype).to(self.device).eval()

    def detect(self, image_rgb, prompt):
        inputs = self._processor(text=TASK + prompt, images=image_rgb,
                                 return_tensors='pt').to(self.device, self._dtype)
        with self._torch.inference_mode():
            ids = self._model.generate(**inputs, max_new_tokens=128, num_beams=3,
                                       do_sample=False)
        text = self._processor.batch_decode(ids, skip_special_tokens=False)[0]
        h, w = image_rgb.shape[:2]
        parsed = self._processor.post_process_generation(text, task=TASK, image_size=(w, h))
        result = parsed.get(TASK, {})
        return [Detection(tuple(np.asarray(b, dtype=float)), 1.0, prompt)
                for b in result.get('bboxes', [])]
