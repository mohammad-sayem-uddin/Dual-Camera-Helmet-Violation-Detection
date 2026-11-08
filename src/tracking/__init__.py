"""Per-camera hierarchical motorcycle and rider tracking package."""
from src.tracking.camera_tracker import (
    CameraTrackingPipeline,
    MotoTrack,
    OrphanRiderTrack,
    Phase10AdapterConfig,
    RiderSubTrack,
)
from src.tracking.kalman_bytetrack import (
    ByteTrackConfig,
    KalmanBoxFilter,
    TrackLifecycleState,
    bbox_iou_xyxy,
    hungarian_iou_match,
    two_stage_bytetrack_assign,
)
from src.tracking.moto_rider_coupler import (
    CoupledMotorcycleObservation,
    CoupledRiderObservation,
    CouplerConfig,
    CouplingGeometricFeatures,
    FrameCouplingResult,
    MotoRiderCoupler,
    OrphanRiderObservation,
)

__all__ = [
    "ByteTrackConfig",
    "CameraTrackingPipeline",
    "CoupledMotorcycleObservation",
    "CoupledRiderObservation",
    "CouplerConfig",
    "CouplingGeometricFeatures",
    "FrameCouplingResult",
    "KalmanBoxFilter",
    "MotoRiderCoupler",
    "MotoTrack",
    "OrphanRiderObservation",
    "OrphanRiderTrack",
    "Phase10AdapterConfig",
    "RiderSubTrack",
    "TrackLifecycleState",
    "bbox_iou_xyxy",
    "hungarian_iou_match",
    "two_stage_bytetrack_assign",
]
