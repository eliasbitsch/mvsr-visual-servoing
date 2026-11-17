import os
import subprocess
import sys

from vs_perception.detectors.base import Detection, Detector, register_detector
from vs_perception.detectors.locate_anything_worker import read_msg, write_msg

# transformers 4.57.1, installed there by docker/Dockerfile.fm
WORKER_SITE = os.environ.get('VS_LA_SITE', '/opt/la-site')


@register_detector('locate_anything')
class LocateAnything(Detector):
    """NVIDIA LocateAnything-3B (NVIDIA licence, non-commercial research), phrase grounding.

    Runs in a worker process (locate_anything_worker.py) because its remote code only works
    with transformers 4.57. No calibrated confidence: boxes are ranked in the order it names them.
    """

    def load(self):
        model_id = self.options.get('model_id', 'nvidia/LocateAnything-3B')
        env = {**os.environ,
               'PYTHONPATH': os.pathsep.join(p for p in (WORKER_SITE,
                                                         os.environ.get('PYTHONPATH')) if p)}
        self._proc = subprocess.Popen(
            [sys.executable, '-m', 'vs_perception.detectors.locate_anything_worker',
             model_id, self.device], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
        if read_msg(self._proc.stdout) != 'ready':
            raise RuntimeError(f'LocateAnything worker exited with {self._proc.wait()}')

    def detect(self, image_rgb, prompt):
        write_msg(self._proc.stdin, (image_rgb, prompt))
        reply = read_msg(self._proc.stdout)
        if reply is None:
            raise RuntimeError('LocateAnything worker died')
        status, payload = reply
        if status != 'ok':
            raise RuntimeError(payload)
        # No confidence: the first box it names is its best guess, later ones rank lower
        return [Detection(box, 1.0 - 0.01 * i, prompt) for i, box in enumerate(payload)]

    def __del__(self):
        proc = getattr(self, '_proc', None)
        if proc and proc.poll() is None:
            proc.kill()
