"""Per-Camera Structured Detection Pipeline (COCO Vehicle/Person + Phase 9 Helmet).

Executes dual-model inference for an individual camera stream (`front` or `back`)
and outputs strongly typed `StructuredDetection` and `FrameDetectionBundle` records.

Enforces strict semantic separation:
- `yolov8n.pt` (COCO) produces ONLY `motorcycle` (class 3) and `person` (class 0).
- `best.pt` (Phase 9) produces ONLY cranial `no_helmet` (class 0) and `helmet` (class 1).
- A helmet/no_helmet detection is NEVER treated as a full-body person detection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from src.models.helmet_detector_adapter import HelmetDetectorAdapter
from src.models.vehicle_detector_adapter import VehiclePersonDetectorAdapter


@dataclass(frozen=True)
class StructuredDetection:
    """Strongly typed bounding box detection with source model and PTS provenance."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    category: str  # "motorcycle", "person", or "head"
    source_model: str  # "yolov8n_coco" or "phase9_best_helmet"
    camera_id: str  # "front" or "back"
    frame_index: int
    pts_timestamp_sec: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)

    def iou(self, other: StructuredDetection | tuple[float, float, float, float]) -> float:
        """Compute Intersection over Union (IoU) with another detection or (x1, y1, x2, y2) tuple."""
        ox1, oy1, ox2, oy2 = other.bbox if isinstance(other, StructuredDetection) else other
        ix1 = max(self.x1, ox1)
        iy1 = max(self.y1, oy1)
        ix2 = min(self.x2, ox2)
        iy2 = min(self.y2, oy2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        other_area = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)
        union = self.area + other_area - inter
        return inter / union if union > 1e-8 else 0.0

    def intersection_over_self_area(
        self, other: StructuredDetection | tuple[float, float, float, float]
    ) -> float:
        """Compute fraction of THIS detection's area contained inside `other` (IoA)."""
        if self.area <= 1e-8:
            return 0.0
        ox1, oy1, ox2, oy2 = other.bbox if isinstance(other, StructuredDetection) else other
        ix1 = max(self.x1, ox1)
        iy1 = max(self.y1, oy1)
        ix2 = min(self.x2, ox2)
        iy2 = min(self.y2, oy2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        return inter / self.area

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": [round(self.x1, 2), round(self.y1, 2), round(self.x2, 2), round(self.y2, 2)],
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "category": self.category,
            "source_model": self.source_model,
            "camera_id": self.camera_id,
            "frame_index": self.frame_index,
            "pts_timestamp_sec": round(self.pts_timestamp_sec, 6),
        }


@dataclass(frozen=True)
class FrameDetectionBundle:
    """Complete structured detections for a single physical camera frame."""
    camera_id: str
    frame_index: int
    pts_timestamp_sec: float
    motorcycles: list[StructuredDetection]
    persons: list[StructuredDetection]
    helmet_boxes: list[StructuredDetection]
    no_helmet_boxes: list[StructuredDetection]
    all_heads: list[StructuredDetection]
    vehicle_latency_ms: float
    helmet_latency_ms: float
    total_latency_ms: float
    device: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "frame_index": self.frame_index,
            "pts_timestamp_sec": round(self.pts_timestamp_sec, 6),
            "num_motorcycles": len(self.motorcycles),
            "num_persons": len(self.persons),
            "num_helmet": len(self.helmet_boxes),
            "num_no_helmet": len(self.no_helmet_boxes),
            "vehicle_latency_ms": round(self.vehicle_latency_ms, 2),
            "helmet_latency_ms": round(self.helmet_latency_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "device": self.device,
            "motorcycles": [d.to_dict() for d in self.motorcycles],
            "persons": [d.to_dict() for d in self.persons],
            "heads": [d.to_dict() for d in self.all_heads],
        }


class SharedDetectorRegistry:
    """
    Lazy singleton cache for YOLO models so Front and Back camera pipelines share
    a single in-memory instance of `yolov8n.pt` and `best.pt` on Apple M1 (8 GB RAM).
    """

    _vehicle_detectors: dict[str, VehiclePersonDetectorAdapter] = {}
    _helmet_detectors: dict[str, HelmetDetectorAdapter] = {}

    @classmethod
    def get_vehicle_detector(
        cls,
        device: str | None = None,
        conf_threshold: float = 0.18,
    ) -> VehiclePersonDetectorAdapter:
        key = f"{device or 'auto'}"
        if key not in cls._vehicle_detectors:
            cls._vehicle_detectors[key] = VehiclePersonDetectorAdapter(
                device=device,
                confidence_threshold=conf_threshold,
            )
        return cls._vehicle_detectors[key]

    @classmethod
    def get_helmet_detector(
        cls,
        device: str | None = None,
        conf_threshold: float = 0.25,
    ) -> HelmetDetectorAdapter:
        key = f"{device or 'auto'}"
        if key not in cls._helmet_detectors:
            cls._helmet_detectors[key] = HelmetDetectorAdapter(
                device=device,
                confidence_threshold=conf_threshold,
            )
        return cls._helmet_detectors[key]


class PerCameraDetector:
    """
    Independent per-camera detection wrapper executing COCO `yolov8n.pt` (motorcycle/person)
    and Phase 9 `best.pt` (helmet/no_helmet) on unique physical camera frames.
    """

    def __init__(
        self,
        camera_id: str,
        device: str | None = None,
        vehicle_conf_threshold: float = 0.18,
        helmet_conf_threshold: float = 0.25,
        vehicle_detector: VehiclePersonDetectorAdapter | None = None,
        helmet_detector: HelmetDetectorAdapter | None = None,
    ) -> None:
        if camera_id not in ("front", "back"):
            raise ValueError(f"camera_id must be 'front' or 'back', got '{camera_id}'")
        self.camera_id = camera_id
        self.vehicle_conf_threshold = float(vehicle_conf_threshold)
        self.helmet_conf_threshold = float(helmet_conf_threshold)

        self.vehicle_detector = vehicle_detector or SharedDetectorRegistry.get_vehicle_detector(
            device=device, conf_threshold=vehicle_conf_threshold
        )
        self.helmet_detector = helmet_detector or SharedDetectorRegistry.get_helmet_detector(
            device=device, conf_threshold=helmet_conf_threshold
        )

        # Verify Phase 9 helmet model classes strictly equal {0: 'no_helmet', 1: 'helmet'}
        expected_helmet_classes = {0: "no_helmet", 1: "helmet"}
        if dict(self.helmet_detector.class_names) != expected_helmet_classes:
            raise ValueError(
                f"Helmet detector classes {self.helmet_detector.class_names} do not match "
                f"expected {expected_helmet_classes}"
            )

    def detect_frame(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        pts_timestamp_sec: float,
    ) -> FrameDetectionBundle:
        """Run both detectors on a single physical frame and package structured detections."""
        veh_res = self.vehicle_detector.predict(
            frame_bgr, conf_threshold=self.vehicle_conf_threshold
        )
        helm_res = self.helmet_detector.predict(
            frame_bgr, conf_threshold=self.helmet_conf_threshold
        )

        motorcycles: list[StructuredDetection] = []
        persons: list[StructuredDetection] = []
        helmet_boxes: list[StructuredDetection] = []
        no_helmet_boxes: list[StructuredDetection] = []
        all_heads: list[StructuredDetection] = []

        for d in veh_res["detections"]:
            x1, y1, x2, y2 = d["bbox"]
            cid = int(d["class_id"])
            cname = str(d["class_name"])
            cat = "motorcycle" if cid == 3 else "person"
            sd = StructuredDetection(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                confidence=float(d["confidence"]),
                class_id=cid,
                class_name=cname,
                category=cat,
                source_model="yolov8n_coco",
                camera_id=self.camera_id,
                frame_index=int(frame_index),
                pts_timestamp_sec=float(pts_timestamp_sec),
            )
            if cid == 3:
                motorcycles.append(sd)
            elif cid == 0:
                persons.append(sd)

        for d in helm_res["detections"]:
            x1, y1, x2, y2 = d["bbox"]
            cid = int(d["class_id"])
            cname = str(d["class_name"])
            sd = StructuredDetection(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                confidence=float(d["confidence"]),
                class_id=cid,
                class_name=cname,
                category="head",
                source_model="phase9_best_helmet",
                camera_id=self.camera_id,
                frame_index=int(frame_index),
                pts_timestamp_sec=float(pts_timestamp_sec),
            )
            all_heads.append(sd)
            if cid == 1:
                helmet_boxes.append(sd)
            else:
                no_helmet_boxes.append(sd)

        v_lat = float(veh_res["latency_ms"])
        h_lat = float(helm_res["latency_ms"])

        return FrameDetectionBundle(
            camera_id=self.camera_id,
            frame_index=int(frame_index),
            pts_timestamp_sec=float(pts_timestamp_sec),
            motorcycles=motorcycles,
            persons=persons,
            helmet_boxes=helmet_boxes,
            no_helmet_boxes=no_helmet_boxes,
            all_heads=all_heads,
            vehicle_latency_ms=v_lat,
            helmet_latency_ms=h_lat,
            total_latency_ms=round(v_lat + h_lat, 2),
            device=self.vehicle_detector.device,
        )
