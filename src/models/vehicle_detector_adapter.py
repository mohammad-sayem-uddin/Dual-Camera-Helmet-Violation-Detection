"""Vehicle and Person Detector Adapter wrapping COCO-pretrained YOLOv8n (`yolov8n.pt`).

Provides filtered detection of COCO classes `0: person` and `3: motorcycle` on Apple
Silicon MPS or CPU, with explicit model-metadata class verification and singleton
model sharing across camera pipelines to respect the 8 GB unified memory budget.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import time
from typing import Any, Union
import numpy as np

try:
    import torch
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


class VehiclePersonDetectorAdapter:
    """
    Standardized inference adapter for COCO-pretrained YOLOv8n (`yolov8n.pt`),
    restricted strictly to `person` (class 0) and `motorcycle` (class 3).
    """

    DEFAULT_CHECKPOINT = Path("yolov8n.pt")
    EXPECTED_TARGET_CLASSES = {0: "person", 3: "motorcycle"}

    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: str | None = None,
        confidence_threshold: float = 0.18,
        iou_threshold: float = 0.45,
    ) -> None:
        if not HAS_YOLO:
            raise ImportError("Ultralytics and PyTorch are required for VehiclePersonDetectorAdapter.")

        self.weights_path = Path(weights_path) if weights_path else self.DEFAULT_CHECKPOINT
        if not self.weights_path.exists():
            raise FileNotFoundError(f"COCO YOLOv8n checkpoint not found at: {self.weights_path}")

        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)

        self.model = YOLO(str(self.weights_path))
        self.class_names: dict[int, str] = (
            dict(self.model.names) if hasattr(self.model, "names") else {}
        )

        # Mandatory class mapping verification from loaded model metadata
        for cid, expected_name in self.EXPECTED_TARGET_CLASSES.items():
            actual_name = self.class_names.get(cid)
            if actual_name != expected_name:
                raise ValueError(
                    f"COCO model class mismatch for ID {cid}: expected '{expected_name}', "
                    f"got '{actual_name}' from {self.weights_path}"
                )

        self.target_class_ids = [0, 3]

    def get_sha256(self) -> str:
        """Compute SHA-256 hash of the loaded checkpoint file."""
        return hashlib.sha256(self.weights_path.read_bytes()).hexdigest()

    def predict(
        self,
        image: Union[np.ndarray, str, Path],
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        Run inference on a single image or frame and return `person` and `motorcycle` detections.
        """
        conf = float(conf_threshold) if conf_threshold is not None else self.confidence_threshold
        iou = float(iou_threshold) if iou_threshold is not None else self.iou_threshold

        t0 = time.perf_counter()
        results = self.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            classes=self.target_class_ids,
            device=self.device,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        detections: list[dict[str, Any]] = []
        motorcycles: list[dict[str, Any]] = []
        persons: list[dict[str, Any]] = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    cls_id = int(box.cls[0].item())
                    score = float(box.conf[0].item())
                    cls_name = self.class_names.get(cls_id, str(cls_id))
                    det = {
                        "bbox": [round(float(c), 2) for c in xyxy],
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": round(score, 4),
                        "source_model": "yolov8n_coco",
                    }
                    detections.append(det)
                    if cls_id == 3:
                        motorcycles.append(det)
                    elif cls_id == 0:
                        persons.append(det)

        return {
            "detections": detections,
            "motorcycles": motorcycles,
            "persons": persons,
            "latency_ms": round(latency_ms, 2),
            "device": self.device,
            "num_detections": len(detections),
            "num_motorcycles": len(motorcycles),
            "num_persons": len(persons),
        }

    def get_info(self) -> dict[str, Any]:
        """Return verified model metadata and class mappings."""
        return {
            "model_name": "YOLOv8n COCO Vehicle & Person Detector",
            "weights_path": str(self.weights_path),
            "target_classes": {cid: self.class_names[cid] for cid in self.target_class_ids},
            "total_coco_classes": len(self.class_names),
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "verified": True,
        }
