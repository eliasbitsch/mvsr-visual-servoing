# vs_perception

Localises the target described by a text prompt and publishes image features on `/vs/features`
(`vs_interfaces/ObjectFeatures`). Three exchangeable parts, each picked by a ROS parameter:

| Part | Parameter | Available | Runs |
|---|---|---|---|
| Detector (foundation model) | `detector` | `grounding_dino`, `owlv2`, `yolo_world`, `ollama_vlm` | worker thread, every `detector_period` s |
| Tracker | `tracker` | `csrt`, `kcf`, `none` | every camera frame |
| Mask refiner | `refiner` | `none`, `sam2` | every camera frame, on the tracked box |

When a detection arrives it belongs to an old frame. The node re-initialises the tracker on that
buffered frame and replays all frames since, so the detector latency does not cause a jump.
`tracker:=none` is the baseline without tracking: features only exist when a detection arrives.

## Adding a foundation model

```python
# vs_perception/detectors/my_model.py
from vs_perception.detectors.base import Detection, Detector, register_detector


@register_detector('my_model')
class MyModel(Detector):

    def load(self):
        import torch                       # heavy imports here, not at module level
        self._model = ...                  # self.device, self.options are available

    def detect(self, image_rgb, prompt):   # HxWx3 uint8, RGB
        return [Detection(box=(x0, y0, x1, y1), score=0.9, label=prompt)]
```

Then add `my_model` to the import line in `detectors/__init__.py`. Nothing else changes.
Extra options (model id, weights) arrive as keyword arguments in `self.options`.

## Testing a model on images (no simulation needed)

The perception code runs in the venv interpreter `$VS_PYTHON` (numpy 2, OpenCV 4.12, PyTorch),
so `ros2 run` needs `--prefix`; `servo.launch.py` does this automatically.

```bash
ros2 run --prefix "$VS_PYTHON" vs_perception detect_image --detector owlv2 \
    --prompt "red box" --prompt "a small red cube" /data/frames/*.png --out /data/results/owlv2
```

Writes annotated images and `detections.csv` with boxes, scores and latency per prompt.

## Runtime changes

```bash
ros2 param set /perception_node prompt "blue cylinder"
ros2 param set /perception_node detector_period 0.2
ros2 run rqt_image_view rqt_image_view /vs/debug_image
```
