"""Vision-language models served by Ollama (Qwen3-VL, Gemma 4, ...) as open-vocabulary detector.

Talks HTTP only, so the model can run in the compose service `ollama` or in any other Ollama
instance (VS_OLLAMA_URL). VLMs return no calibrated confidence, every box gets score 1.0.
"""
import base64
import json
import os
import re
import urllib.request

import cv2

from vs_perception.detectors.base import Detection, Detector, register_detector

# Box conventions the models were trained on, all normalised to 0..1000
FORMATS = {
    # Qwen3-VL: {"bbox_2d": [x0, y0, x1, y1], "label": ...}
    'qwen': ('bbox_2d', (0, 1, 2, 3),
             'Locate every {prompt} in the image. Output JSON: '
             '[{{"bbox_2d": [x1, y1, x2, y2], "label": "{prompt}"}}]. Output [] if there is none.'),
    # Gemma: {"box_2d": [y0, x0, y1, x1], "label": ...}
    'gemma': ('box_2d', (1, 0, 3, 2),
              'Detect every {prompt} in the image. Output a JSON list with "box_2d" '
              '([ymin, xmin, ymax, xmax], normalised to 0-1000) and "label". '
              'Output [] if there is none.'),
}


def _box_format(model):
    return 'gemma' if 'gemma' in model.lower() else 'qwen'


def _parse_json_list(text):
    """VLMs wrap JSON in prose or ``` fences; take the outermost list or object."""
    match = re.search(r'\[.*\]|\{.*\}', text, re.S)
    if not match:
        return []
    data = json.loads(match.group(0))
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), [data])
    return [d for d in data if isinstance(d, dict)]


@register_detector('ollama_vlm')
class OllamaVlm(Detector):

    def load(self):
        self.url = self.options.get('url', os.environ.get('VS_OLLAMA_URL', 'http://localhost:11434'))
        self.model = self.options.get('model', os.environ.get('VS_OLLAMA_MODEL',
                                                              'qwen3-vl:8b-instruct'))
        self.format = self.options.get('box_format', _box_format(self.model))
        self.timeout = float(self.options.get('timeout', 60))
        # Fail early with a clear message instead of on the first frame
        with urllib.request.urlopen(f'{self.url}/api/tags', timeout=5) as resp:
            names = [m['name'] for m in json.load(resp)['models']]
        if self.model not in names:
            raise RuntimeError(f'{self.model} not in Ollama at {self.url}, have: {names}. '
                               f'Run: ollama pull {self.model}')

    def detect(self, image_rgb, prompt):
        key, order, template = FORMATS[self.format]
        ok, jpg = cv2.imencode('.jpg', image_rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
        body = {
            'model': self.model,
            'stream': False,
            'think': False,
            'keep_alive': '30m',
            'options': {'temperature': 0},
            'messages': [{'role': 'user', 'content': template.format(prompt=prompt),
                          'images': [base64.b64encode(jpg.tobytes()).decode()]}],
        }
        req = urllib.request.Request(f'{self.url}/api/chat', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            text = json.load(resp)['message']['content']

        h, w = image_rgb.shape[:2]
        dets = []
        for item in _parse_json_list(text):
            raw = item.get(key) or item.get('bbox_2d') or item.get('box_2d')
            if not isinstance(raw, list) or len(raw) != 4:
                continue
            x0, y0, x1, y1 = (float(raw[i]) for i in order)
            box = (x0 / 1000 * w, y0 / 1000 * h, x1 / 1000 * w, y1 / 1000 * h)
            if box[2] > box[0] and box[3] > box[1]:
                dets.append(Detection(box, 1.0, str(item.get('label', prompt))))
        return dets
