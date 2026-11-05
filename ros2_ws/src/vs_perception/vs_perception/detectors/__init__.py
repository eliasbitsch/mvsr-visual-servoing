"""Importing a module registers its detector. New models: add the import here."""
from vs_perception.detectors import grounding_dino, owlv2, vlm, yolo_world  # noqa: F401
from vs_perception.detectors.base import DETECTORS, Detection, Detector, register_detector

__all__ = ['DETECTORS', 'Detection', 'Detector', 'register_detector']
