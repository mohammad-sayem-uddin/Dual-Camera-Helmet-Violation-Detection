"""Synthetic Scenario Generator and Controlled Event Evaluator for Temporal Verification."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.evaluation.temporal_metrics import TemporalMetrics, compute_temporal_metrics
from src.temporal.rider_tracker import BoundingBox
from src.temporal.state_machine import TemporalState
from src.temporal.temporal_engine import TemporalEngineConfig, TemporalVerificationEngine


def generate_synthetic_scenarios() -> dict[str, dict[str, Any]]:
    """
    Constructs standardized benchmark scenarios representing critical traffic conditions.
    
    Scenarios:
    1. stable_helmet: Compliant rider moving across 30 frames (c = 0.90..0.96)
    2. stable_violation: Non-compliant rider moving across 30 frames (c = 0.85..0.95)
    3. fp_burst_on_compliant: Helmeted rider with 2 spurious false-positive no_helmet frames
    4. short_occlusion: Non-compliant rider occluded for 4 frames (< N_grace)
    5. long_occlusion: Rider occluded for 18 frames (> N_lost)
    6. marginal_jitter: Ambiguous detections near detection boundary (c = 0.28..0.34)
    7. multi_rider_crossing: Two simultaneous riders with overlapping trajectory ranges
    """
    scenarios: dict[str, dict[str, Any]] = {}

    # 1. Stable Helmet (30 frames)
    frames_s1 = []
    for f in range(30):
        x = 100.0 + f * 5.0
        frames_s1.append([
            {"bbox": [x, 150.0, x + 60.0, 230.0], "confidence": 0.92, "class_id": 1, "class_name": "helmet"}
        ])
    scenarios["stable_helmet"] = {
        "description": "Consistent helmeted rider (30 frames)",
        "ground_truth": {1: "helmet"},
        "frames": frames_s1,
    }

    # 2. Stable Violation (30 frames)
    frames_s2 = []
    for f in range(30):
        x = 120.0 + f * 5.0
        frames_s2.append([
            {"bbox": [x, 160.0, x + 55.0, 235.0], "confidence": 0.90, "class_id": 0, "class_name": "no_helmet"}
        ])
    scenarios["stable_violation"] = {
        "description": "Consistent non-helmeted rider (30 frames)",
        "ground_truth": {1: "no_helmet"},
        "frames": frames_s2,
    }

    # 3. FP Burst on Compliant Rider (30 frames with 2 spurious no_helmet frames at f=10, f=11)
    frames_s3 = []
    for f in range(30):
        x = 100.0 + f * 5.0
        if f in (10, 11):
            frames_s3.append([
                {"bbox": [x, 150.0, x + 60.0, 230.0], "confidence": 0.55, "class_id": 0, "class_name": "no_helmet"}
            ])
        else:
            frames_s3.append([
                {"bbox": [x, 150.0, x + 60.0, 230.0], "confidence": 0.94, "class_id": 1, "class_name": "helmet"}
            ])
    scenarios["fp_burst_on_compliant"] = {
        "description": "Helmeted rider subjected to 2-frame noise burst of false no_helmet",
        "ground_truth": {1: "helmet"},
        "frames": frames_s3,
    }

    # 4. Short Occlusion (Violation rider occluded at f=12..15, then resumes)
    frames_s4 = []
    for f in range(30):
        x = 100.0 + f * 5.0
        if 12 <= f <= 15:
            frames_s4.append([])  # Occluded by obstacle / signpost
        else:
            frames_s4.append([
                {"bbox": [x, 150.0, x + 60.0, 230.0], "confidence": 0.88, "class_id": 0, "class_name": "no_helmet"}
            ])
    scenarios["short_occlusion"] = {
        "description": "Violation rider occluded for 4 frames (< N_grace) and re-acquired",
        "ground_truth": {1: "no_helmet"},
        "frames": frames_s4,
    }

    # 5. Long Occlusion (Rider disappears for 18 frames > N_lost)
    frames_s5 = []
    for f in range(30):
        x = 100.0 + f * 5.0
        if f >= 10:
            frames_s5.append([])  # Disappears completely
        else:
            frames_s5.append([
                {"bbox": [x, 150.0, x + 60.0, 230.0], "confidence": 0.89, "class_id": 1, "class_name": "helmet"}
            ])
    scenarios["long_occlusion"] = {
        "description": "Rider track permanently lost after extended occlusion (> N_lost)",
        "ground_truth": {1: "helmet"},
        "frames": frames_s5,
    }

    # 6. Marginal Jitter (Low-confidence alternating detections near threshold)
    frames_s6 = []
    for f in range(25):
        x = 100.0 + f * 4.0
        cls_name = "no_helmet" if f % 2 == 0 else "helmet"
        cls_id = 0 if cls_name == "no_helmet" else 1
        frames_s6.append([
            {"bbox": [x, 150.0, x + 50.0, 220.0], "confidence": 0.30, "class_id": cls_id, "class_name": cls_name}
        ])
    scenarios["marginal_jitter"] = {
        "description": "Ambiguous low-confidence alternating predictions (c = 0.30)",
        "ground_truth": {1: "unknown"},
        "frames": frames_s6,
    }

    # 7. Multi-Rider Scene (Two simultaneous motorcycles: Track 1 = Helmet, Track 2 = Violation)
    frames_s7 = []
    for f in range(30):
        x1 = 80.0 + f * 6.0
        x2 = 250.0 + f * 5.0
        frames_s7.append([
            {"bbox": [x1, 140.0, x1 + 55.0, 220.0], "confidence": 0.93, "class_id": 1, "class_name": "helmet"},
            {"bbox": [x2, 180.0, x2 + 60.0, 260.0], "confidence": 0.89, "class_id": 0, "class_name": "no_helmet"},
        ])
    scenarios["multi_rider_crossing"] = {
        "description": "Two simultaneous riders: one compliant helmeted, one non-helmeted violation",
        "ground_truth": {1: "helmet", 2: "no_helmet"},
        "frames": frames_s7,
    }

    return scenarios


def evaluate_scenario(
    scenario_data: dict[str, Any],
    aggregator_method: str = "confidence_weighted",
    window_size: int = 10,
    fps: float = 30.0,
) -> dict[str, Any]:
    """Runs a single synthetic scenario through the temporal verification engine."""
    config = TemporalEngineConfig(
        aggregator_method=aggregator_method,
        sliding_window_size=window_size,
    )
    engine = TemporalVerificationEngine(config=config)

    frames = scenario_data["frames"]
    gt = scenario_data["ground_truth"]

    for f_idx, raw_dets in enumerate(frames):
        engine.process_frame(frame_index=f_idx, raw_detections=raw_dets, fps=fps)

    all_tracks = list(engine.tracker.active_tracks.values()) + engine.tracker.dead_tracks
    metrics = compute_temporal_metrics(evaluated_tracks=all_tracks, ground_truth_labels=gt, fps=fps)

    return {
        "scenario": scenario_data.get("description", ""),
        "aggregator_method": aggregator_method,
        "metrics": metrics.to_dict(),
        "summary": engine.get_summary(),
        "alerts": [a.to_dict() for a in engine.decision_engine.alerts_log],
    }


def run_full_synthetic_benchmark() -> dict[str, Any]:
    """
    Executes all synthetic benchmark scenarios across all 3 aggregation methods:
    1. simple_voting
    2. confidence_weighted
    3. ema
    """
    scenarios = generate_synthetic_scenarios()
    methods = ["simple_voting", "confidence_weighted", "ema"]
    benchmark_results: dict[str, Any] = {}

    for method in methods:
        benchmark_results[method] = {}
        for s_name, s_data in scenarios.items():
            res = evaluate_scenario(scenario_data=s_data, aggregator_method=method)
            benchmark_results[method][s_name] = res

    return benchmark_results
