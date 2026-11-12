"""Dual-camera evidence fusion package (`src/fusion/`, TASK-11)."""
from src.fusion.dual_camera_fusion import (
    DualCameraFusionEngine,
    OrphanFusionResult,
    PairFusionOutput,
    derive_event_type_from_fused_riders,
    extract_camera_moto_and_orphan_records,
)
from src.fusion.evidence_fusion import (
    FusionConfig,
    FusionDecisionState,
    RiderFusionResult,
    ViewPosteriorSummary,
    ViewRiderEvidence,
    compute_contradiction_metric,
    compute_mean_head_area_from_history,
    compute_sync_reliability,
    compute_view_posterior_and_weight,
    evaluate_single_view_belief,
    fuse_rider_beliefs,
)

__all__ = [
    "DualCameraFusionEngine",
    "FusionConfig",
    "FusionDecisionState",
    "OrphanFusionResult",
    "PairFusionOutput",
    "RiderFusionResult",
    "ViewPosteriorSummary",
    "ViewRiderEvidence",
    "compute_contradiction_metric",
    "compute_mean_head_area_from_history",
    "compute_sync_reliability",
    "compute_view_posterior_and_weight",
    "derive_event_type_from_fused_riders",
    "evaluate_single_view_belief",
    "extract_camera_moto_and_orphan_records",
    "fuse_rider_beliefs",
]
