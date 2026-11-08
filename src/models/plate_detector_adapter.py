"""Plate Detector Adapter Interface and Heuristic Extractor.
Defines modular contract for license plate localization on motorcycle video frames.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BasePlateDetectorAdapter(ABC):
    """Abstract base adapter for motorcycle license plate detection."""

    @abstractmethod
    def detect_plates(
        self,
        frame: np.ndarray,
        vehicle_bbox: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Detect license plates in a full frame or within a vehicle bounding box.
        
        Args:
            frame: Full BGR image array.
            vehicle_bbox: Optional [x1, y1, x2, y2] bounding box of the motorcycle.
            
        Returns:
            List of detected plates: [{'bbox': [x1, y1, x2, y2], 'confidence': float, 'crop': np.ndarray}]
        """
        pass


class HeuristicPlateDetector(BasePlateDetectorAdapter):
    """
    Heuristic plate region extractor for rear motorcycle views.
    In the absence of a dedicated trained plate detector, extracts the lower-center
    region of the motorcycle rear track where license plates are standardly mounted.
    """

    def __init__(
        self,
        vertical_start_ratio: float = 0.65,
        vertical_end_ratio: float = 0.95,
        horizontal_margin_ratio: float = 0.20,
    ) -> None:
        self.v_start = vertical_start_ratio
        self.v_end = vertical_end_ratio
        self.h_margin = horizontal_margin_ratio

    def detect_plates(
        self,
        frame: np.ndarray,
        vehicle_bbox: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        if vehicle_bbox is None or frame is None or frame.size == 0:
            return []

        x1, y1, x2, y2 = vehicle_bbox
        w = x2 - x1
        h = y2 - y1

        if w <= 10 or h <= 10:
            return []

        # Plate ROI in lower central region of the rear motorcycle bounding box
        p_x1 = max(0, int(x1 + w * self.h_margin))
        p_x2 = min(frame.shape[1], int(x2 - w * self.h_margin))
        p_y1 = max(0, int(y1 + h * self.v_start))
        p_y2 = min(frame.shape[0], int(y1 + h * self.v_end))

        if p_x2 <= p_x1 or p_y2 <= p_y1:
            return []

        crop = frame[p_y1:p_y2, p_x1:p_x2]

        return [{
            "bbox": [float(p_x1), float(p_y1), float(p_x2), float(p_y2)],
            "confidence": 0.50,  # Heuristic prior confidence
            "crop": crop,
            "method": "HEURISTIC_REAR_MOTORCYCLE_ROI",
        }]


class PretrainedPlateDetectorAdapter(BasePlateDetectorAdapter):
    """
    Interface placeholder for an external pretrained license plate detector (e.g. YOLOv8-plate).
    Status: INTERFACE_DEFINED_NOT_TRAINED.
    To be populated by Gemini 4 Argon if a dedicated pretrained plate detector is selected.
    """

    def __init__(self, weights_path: str | None = None) -> None:
        self.weights_path = weights_path
        self.status = "INTERFACE_DEFINED_AWAITING_WEIGHTS"

    def detect_plates(
        self,
        frame: np.ndarray,
        vehicle_bbox: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError(
            "PretrainedPlateDetectorAdapter is an interface definition. "
            "Use HeuristicPlateDetector or provide a trained plate detection checkpoint in Phase 12."
        )
