"""Model adapters package for helmet detection, license plate detection, and OCR."""
from __future__ import annotations

from src.models.helmet_detector_adapter import HelmetDetectorAdapter
from src.models.plate_detector_adapter import BasePlateDetectorAdapter, HeuristicPlateDetector
from src.models.ocr_adapter import BaseOCRAdapter, EasyOCRAdapter

__all__ = [
    "HelmetDetectorAdapter",
    "BasePlateDetectorAdapter",
    "HeuristicPlateDetector",
    "BaseOCRAdapter",
    "EasyOCRAdapter",
]
