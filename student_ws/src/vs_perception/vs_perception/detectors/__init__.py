"""Importing a module registers its detector. New models: add the import here."""
from vs_perception.detectors import (florence2, grounding_dino, locate_anything, owlv2,  # noqa: F401
                                     sam3, vlm, yolo_world, yoloe)
from vs_perception.detectors.base import DETECTORS, Detection, Detector, register_detector

__all__ = ['DETECTORS', 'Detection', 'Detector', 'register_detector']
