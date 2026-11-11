"""Vision-language models (Qwen3-VL, Gemma 4, ...) as open-vocabulary detector.

Uses the OpenAI-compatible chat API, so the server is exchangeable: Ollama (default, compose
profile `vlm`), vLLM, llama.cpp server, SGLang or LM Studio. VLMs return no calibrated
confidence, every box gets score 1.0.
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
    # Qwen3-VL / Qwen3.5: {"bbox_2d": [x0, y0, x1, y1], "label": ...}
    'qwen': ('bbox_2d', (0, 1, 2, 3),
             'Locate every {prompt} in the image. Output JSON: '
             '[{{"bbox_2d": [x1, y1, x2, y2], "label": "{prompt}"}}]. Output [] if there is none.'),
    # Gemma: {"box_2d": [y0, x0, y1, x1], "label": ...}
    'gemma': ('box_2d', (1, 0, 3, 2),
              'Detect every {prompt} in the image. Output a JSON list with "box_2d" '
              '([ymin, xmin, ymax, xmax], normalised to 0-1000) and "label". '
              'Output [] if there is none.'),
    # GLM-4.xV: free text with <|begin_of_box|>[x0, y0, x1, y1]<|end_of_box|>
    'glm': (None, (0, 1, 2, 3),
            'Locate every {prompt} in the image and give the bounding box of each one.'),
}
_NUMBER = r'-?\d+(?:\.\d+)?'
_QUAD = re.compile(r'\[?\s*\(?\s*' + r'\s*,\s*'.join([f'({_NUMBER})'] * 4))


def _box_format(model):
    name = model.lower()
    return 'gemma' if 'gemma' in name else 'glm' if 'glm' in name else 'qwen'


def _extract_boxes(text, fmt):
    """Return (four numbers in the model's order, label) pairs from the raw model answer."""
    key = FORMATS[fmt][0]
    if key is None:
        spans = re.findall(r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>', text, re.S) or [text]
        return [(list(map(float, m)), '') for span in spans for m in _QUAD.findall(span)]
    # JSON formats; VLMs wrap them in prose or ``` fences, so take the outermost list or object
    match = re.search(r'\[.*\]|\{.*\}', text, re.S)
    try:
        data = json.loads(match.group(0)) if match else []
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), [data])
    out = []
    for item in data:
        raw = item.get(key) or item.get('bbox_2d') or item.get('box_2d') \
            if isinstance(item, dict) else None
        if isinstance(raw, list) and len(raw) == 4:
            out.append((list(map(float, raw)), str(item.get('label', ''))))
    return out


def _get_json(url, body=None, timeout=5.0):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body else None,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


@register_detector('vlm')
class Vlm(Detector):

    def load(self):
        self.url = self.options.get('url', os.environ.get('VS_VLM_URL',
                                                          'http://localhost:11435/v1'))
        self.model = self.options.get('model', os.environ.get('VS_VLM_MODEL',
                                                              'qwen3-vl:8b-instruct'))
        self.format = self.options.get('box_format', _box_format(self.model))
        self.timeout = float(self.options.get('timeout', 60))
        # Longest image side sent to the model, 0 = full resolution. Fewer pixels -> fewer
        # vision tokens -> faster answer; boxes come back normalised, so nothing else changes.
        self.max_side = int(self.options.get('max_side', os.environ.get('VS_VLM_MAX_SIDE', 0)))
        # Fail early with a clear message instead of on the first frame
        names = [m['id'] for m in _get_json(f'{self.url}/models')['data']]
        if self.model not in names:
            raise RuntimeError(f'{self.model} not served at {self.url}, have: {names}')

    def detect(self, image_rgb, prompt):
        _, order, template = FORMATS[self.format]
        sent = image_rgb
        scale = self.max_side / max(image_rgb.shape[:2]) if self.max_side else 1.0
        if scale < 1.0:
            sent = cv2.resize(image_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        _, jpg = cv2.imencode('.jpg', sent[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
        image_url = 'data:image/jpeg;base64,' + base64.b64encode(jpg.tobytes()).decode()
        body = {
            'model': self.model,
            'temperature': 0,
            # GLM reasons before answering and needs room for that
            'max_tokens': int(self.options.get('max_tokens', 2048 if self.format == 'glm' else 512)),
            'messages': [{'role': 'user', 'content': [
                {'type': 'text', 'text': template.format(prompt=prompt)},
                {'type': 'image_url', 'image_url': {'url': image_url}},
            ]}],
        }
        text = _get_json(f'{self.url}/chat/completions', body,
                         self.timeout)['choices'][0]['message']['content']

        h, w = image_rgb.shape[:2]
        dets = []
        for raw, label in _extract_boxes(text, self.format):
            x0, y0, x1, y1 = (raw[i] for i in order)
            box = (x0 / 1000 * w, y0 / 1000 * h, x1 / 1000 * w, y1 / 1000 * h)
            if box[2] > box[0] and box[3] > box[1]:
                dets.append(Detection(box, 1.0, label or prompt))
        return dets
