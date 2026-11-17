"""Device selection for detectors, refiners and trackers: 'auto' picks CUDA when available."""


def resolve_device(device='auto'):
    if device != 'auto':
        return device
    try:
        import torch  # heavy import only when a model is actually created
    except ImportError:
        return 'cpu'
    return 'cuda' if torch.cuda.is_available() else 'cpu'
