"""Name -> class registry, so components are chosen by a ROS parameter string."""
import os


def weights_path(filename):
    """Ultralytics downloads known assets to the given path, so keep them in the model volume."""
    folder = os.environ.get('VS_WEIGHTS_DIR', os.path.expanduser('~/.cache/vs_weights'))
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, filename)


class Registry:

    def __init__(self, kind):
        self.kind = kind
        self._classes = {}

    def register(self, name):
        def decorator(cls):
            if name in self._classes:
                raise ValueError(f'{self.kind} "{name}" registered twice')
            self._classes[name] = cls
            cls.registry_name = name
            return cls
        return decorator

    def create(self, name, **kwargs):
        try:
            cls = self._classes[name]
        except KeyError:
            raise ValueError(
                f'unknown {self.kind} "{name}", available: {", ".join(self.names())}') from None
        return cls(**kwargs)

    def names(self):
        return sorted(self._classes)
