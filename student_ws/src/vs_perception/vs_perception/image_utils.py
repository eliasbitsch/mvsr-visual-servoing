"""sensor_msgs/Image <-> numpy without cv_bridge (cv_bridge is built against the apt OpenCV)."""
import numpy as np
from sensor_msgs.msg import Image


def to_rgb(msg: Image) -> np.ndarray:
    channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}.get(msg.encoding)
    if channels is None:
        raise ValueError(f'unsupported encoding {msg.encoding}')
    rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    img = rows[:, :msg.width * channels].reshape(msg.height, msg.width, channels)[:, :, :3]
    if msg.encoding.startswith('bgr'):
        img = img[:, :, ::-1]
    return np.ascontiguousarray(img)


def from_rgb(img: np.ndarray, header) -> Image:
    msg = Image(header=header, height=img.shape[0], width=img.shape[1], encoding='rgb8',
                is_bigendian=0, step=img.shape[1] * 3)
    msg.data = np.ascontiguousarray(img).tobytes()
    return msg
