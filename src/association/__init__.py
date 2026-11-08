"""Phase 12E Cross-Camera Tracklet Association Package (`src.association`)."""
from src.association.appearance_features import (
    AppearanceConfig,
    bhattacharyya_histogram_similarity,
    compute_crop_quality,
    compute_hsv_achromatic_histogram,
    extract_tracklet_appearance_descriptor,
    score_appearance_similarity,
)
from src.association.association_audit import AssociationAuditor
from src.association.association_features import (
    ConvoyOrderConfig,
    LaneSpatialConfig,
    RiderCardinalityConfig,
    evaluate_convoy_order_cue,
    evaluate_lane_spatial_cue,
    evaluate_rider_cardinality_cue,
)
from src.association.association_scorer import (
    AssociationScorer,
    AssociationScorerConfig,
)
from src.association.candidate_generator import (
    CandidateGenerator,
    CandidatePairEvaluation,
)
from src.association.cross_camera_association import (
    CrossCameraAssociationConfig,
    CrossCameraAssociationEngine,
)
from src.association.global_assignment import (
    GlobalAssignmentConfig,
    GlobalAssignmentSolver,
)
from src.association.orphan_association import (
    OrphanAssociationConfig,
    OrphanAssociationEngine,
)
from src.association.rider_association import (
    RiderAssociationConfig,
    RiderAssociationEngine,
)
from src.association.tracklet_builder import (
    TrackletBuilder,
    TrackletBuilderConfig,
)
from src.association.tracklet_models import (
    AppearanceDescriptorRecord,
    AssociationOutcome,
    CandidatePairScore,
    CrossCameraAssociationBatchResult,
    CueScoreDetail,
    MotorcycleAssociationResult,
    MotorcycleTracklet,
    ObservedFrameRecord,
    OrphanAssociationResult,
    OrphanRiderTracklet,
    RiderAssociationResult,
    RiderTracklet,
    SCHEMA_VERSION,
    TimestampQualityRecord,
    TrackletCompleteness,
)
from src.association.transit_time_model import (
    TransitEvaluationResult,
    TransitTimeConfig,
    TransitTimeModel,
)

__all__ = [
    "AppearanceConfig",
    "AppearanceDescriptorRecord",
    "AssociationAuditor",
    "AssociationOutcome",
    "AssociationScorer",
    "AssociationScorerConfig",
    "CandidateGenerator",
    "CandidatePairEvaluation",
    "CandidatePairScore",
    "ConvoyOrderConfig",
    "CrossCameraAssociationBatchResult",
    "CrossCameraAssociationConfig",
    "CrossCameraAssociationEngine",
    "CueScoreDetail",
    "GlobalAssignmentConfig",
    "GlobalAssignmentSolver",
    "LaneSpatialConfig",
    "MotorcycleAssociationResult",
    "MotorcycleTracklet",
    "ObservedFrameRecord",
    "OrphanAssociationConfig",
    "OrphanAssociationEngine",
    "OrphanAssociationResult",
    "OrphanRiderTracklet",
    "RiderAssociationConfig",
    "RiderAssociationEngine",
    "RiderAssociationResult",
    "RiderCardinalityConfig",
    "RiderTracklet",
    "SCHEMA_VERSION",
    "TimestampQualityRecord",
    "TrackletBuilder",
    "TrackletBuilderConfig",
    "TrackletCompleteness",
    "TransitEvaluationResult",
    "TransitTimeConfig",
    "TransitTimeModel",
    "bhattacharyya_histogram_similarity",
    "compute_crop_quality",
    "compute_hsv_achromatic_histogram",
    "evaluate_convoy_order_cue",
    "evaluate_lane_spatial_cue",
    "evaluate_rider_cardinality_cue",
    "extract_tracklet_appearance_descriptor",
    "score_appearance_similarity",
]
