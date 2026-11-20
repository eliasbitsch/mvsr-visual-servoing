"""Second opinion on detector boxes: does the crop really show what the prompt asks for?

Open-vocabulary detectors and VLMs return a box for prompts of objects that are not in the scene
(the ArUco board as "red can", Gemma 4 E2B on 10 % of the absent-object prompts). The verifier
crops every box and scores it against the prompt with an image-text model; boxes below the
threshold are dropped before the tracker is seeded.
"""
from abc import ABC, abstractmethod

from vs_perception.device import resolve_device
from vs_perception.registry import Registry

VERIFIERS = Registry('verifier')


def crop_with_context(image_rgb, box, context=0.15, min_side=32):
    """Square crop around the box plus some context: the image models expect square inputs and
    would otherwise stretch elongated boxes."""
    h, w = image_rgb.shape[:2]
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0, min_side) * (1 + 2 * context)
    xa, ya = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
    xb, yb = int(min(cx + side / 2, w)), int(min(cy + side / 2, h))
    return image_rgb[ya:max(yb, ya + 1), xa:max(xb, xa + 1)]


class Verifier(ABC):

    def __init__(self, device='auto', threshold=None, **options):
        self.device = resolve_device(device)
        self.options = options
        if threshold is not None:
            self.threshold = float(threshold)

    threshold = 0.0

    def load(self):
        pass

    @abstractmethod
    def scores(self, image_rgb, prompt, boxes):
        """One score in 0..1 per box: how well the crop matches the prompt."""

    def filter(self, image_rgb, prompt, detections):
        """Detections whose crop matches the prompt; the score is stored on each kept one."""
        if not detections:
            return detections
        kept = []
        for det, s in zip(detections, self.scores(image_rgb, prompt, [d.box for d in detections])):
            det.verify_score = float(s)
            if s >= self.threshold:
                kept.append(det)
        return kept


@VERIFIERS.register('none')
class NoVerifier(Verifier):

    def scores(self, image_rgb, prompt, boxes):
        return [1.0] * len(boxes)


@VERIFIERS.register('siglip')
class SiglipVerifier(Verifier):
    """SigLIP 2 (Google, Apache 2.0) via transformers. Its sigmoid loss scores every image-text
    pair on its own, so the score is an absolute "does it match" instead of CLIP's softmax over
    a fixed set of alternatives, which would need a list of everything that might be in view."""

    # Calibrated with vs_evaluation.bench_verifier on the 31 frames of main_dino_sam2 (ground-truth
    # crops): keeps 99 % of the attribute prompts ("red can"), lets 3 % of the other objects' and
    # none of the absent objects' prompts through. Bare categories ("can") score much lower and
    # keep only 57 %: use the verifier with descriptive prompts
    threshold = 0.002

    def load(self):
        import torch
        from transformers import AutoModel, AutoProcessor

        # SigLIP 2: better localisation features than v1, and its Gemma tokenizer needs no
        # sentencepiece (not in the image)
        name = self.options.get('model', 'google/siglip2-base-patch16-224')
        self._torch = torch
        self._dtype = torch.float16 if self.device.startswith('cuda') else torch.float32
        self._processor = AutoProcessor.from_pretrained(name)
        self._model = AutoModel.from_pretrained(name, dtype=self._dtype).to(self.device).eval()
        self._text_cache = {}

    @staticmethod
    def _unit(out):
        # transformers 5 returns an output object with the projected embedding as pooler_output,
        # 4.x the tensor itself
        emb = out if hasattr(out, 'norm') else out.pooler_output
        return emb / emb.norm(dim=-1, keepdim=True)

    def _text(self, prompt):
        # The template SigLIP was evaluated with; lower case as in its training captions
        if prompt not in self._text_cache:
            # SigLIP 2 was trained on text padded to 64 tokens
            inputs = self._processor(text=[f'a photo of a {prompt.lower()}.'],
                                     padding='max_length', max_length=64,
                                     return_tensors='pt').to(self.device)
            with self._torch.no_grad():
                self._text_cache[prompt] = self._unit(self._model.get_text_features(**inputs))
        return self._text_cache[prompt]

    def scores(self, image_rgb, prompt, boxes):
        if not boxes:
            return []
        crops = [crop_with_context(image_rgb, b) for b in boxes]
        inputs = self._processor(images=crops, return_tensors='pt').to(self.device)
        with self._torch.no_grad():
            img = self._unit(self._model.get_image_features(
                pixel_values=inputs['pixel_values'].to(self._dtype)))
            logits = (img @ self._text(prompt).T) * self._model.logit_scale.exp() \
                + self._model.logit_bias
        return self._torch.sigmoid(logits.float()).squeeze(-1).cpu().numpy().tolist()
