"""Helmet Detector Adapter wrapping trained YOLOv8 model for inference on CPU and Apple Silicon MPS."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Union
import numpy as np

try:
    from ultralytics import YOLO
    import torch
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


class HelmetDetectorAdapter:
    """
    Standardized inference adapter for the trained YOLOv8 motorcycle helmet detection model.
    
    Classes:
        0: no_helmet
        1: helmet
    """

    DEFAULT_CHECKPOINT = Path("results/phase9_multiseed/exp_combined/seed_42/weights/best.pt")

    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: str | None = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        if not HAS_YOLO:
            raise ImportError("Ultralytics and PyTorch are required for HelmetDetectorAdapter.")

        self.weights_path = Path(weights_path) if weights_path else self.DEFAULT_CHECKPOINT
        if not self.weights_path.exists():
            # Check fallback in results/exp_combined/weights/best.pt
            fallback = Path("results/exp_combined/weights/best.pt")
            if fallback.exists():
                self.weights_path = fallback
            else:
                raise FileNotFoundError(f"Helmet detector checkpoint not found at: {self.weights_path}")

        # Device selection: cpu, mps, or cuda
        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        # Load YOLO model
        self.model = YOLO(str(self.weights_path))
        self.class_names = self.model.names if hasattr(self.model, "names") else {0: "no_helmet", 1: "helmet"}

    def predict(
        self,
        image: Union[np.ndarray, str, Path],
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        Run inference on a single image or frame.
        
        Args:
            image: BGR numpy array or file path to image.
            conf_threshold: Optional override for confidence threshold.
            iou_threshold: Optional override for NMS IoU threshold.
            
        Returns:
            Dictionary containing:
                - detections: List of dicts with keys: bbox [x1, y1, x2, y2], class_id, class_name, confidence
                - latency_ms: Float inference latency in milliseconds
                - device: Device used
        """
        conf = conf_threshold if conf_threshold is not None else self.confidence_threshold
        iou = iou_threshold if iou_threshold is not None else self.iou_threshold

        t0 = time.perf_counter()
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    cls_id = int(box.cls[0].item())
                    score = float(box.conf[0].item())
                    cls_name = self.class_names.get(cls_id, str(cls_id))
                    detections.append({
                        "bbox": [round(c, 2) for c in xyxy],
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": round(score, 4),
                    })

        return {
            "detections": detections,
            "latency_ms": round(latency_ms, 2),
            "device": self.device,
            "num_detections": len(detections),
        }

    def get_info(self) -> dict[str, Any]:
        """Return model metadata and configuration."""
        return {
            "model_name": "YOLOv8n Helmet Detector",
            "weights_path": str(self.weights_path),
            "classes": self.class_names,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "verified": True,
        }
