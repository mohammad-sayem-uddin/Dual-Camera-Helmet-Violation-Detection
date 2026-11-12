"""Integrated dual-camera pipeline package (Stage G orchestration layer)."""
from src.pipeline.dual_camera_pipeline import (
    CameraRunOutput,
    PairError,
    PairRunResult,
    PairStatus,
    PipelineConfig,
    PipelineError,
    SyncContext,
    derive_pair_events,
    run_all,
    run_pair,
    run_pair_tick_mode,
)

__all__ = [
    "CameraRunOutput",
    "PairError",
    "PairRunResult",
    "PairStatus",
    "PipelineConfig",
    "PipelineError",
    "SyncContext",
    "derive_pair_events",
    "run_all",
    "run_pair",
    "run_pair_tick_mode",
]
