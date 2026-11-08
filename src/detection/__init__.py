"""Per-camera structured object detection package."""
from src.detection.camera_detector import (
    FrameDetectionBundle,
    PerCameraDetector,
    SharedDetectorRegistry,
    StructuredDetection,
)

__all__ = [
    "FrameDetectionBundle",
    "PerCameraDetector",
    "SharedDetectorRegistry",
    "StructuredDetection",
]
