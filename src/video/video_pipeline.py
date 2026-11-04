"""End-to-End Video Verification Pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from src.temporal.temporal_engine import TemporalEngineConfig, TemporalVerificationEngine
from src.video.frame_processor import FrameProcessor


class VideoVerificationPipeline:
    """
    Executes confidence-aware temporal verification across video frame sequences.
    
    Supports:
    - Live or pre-extracted video frame sequences
    - Custom detector callable or offline detection manifests
    - Continuous visualization and alert tracking
    """

    def __init__(
        self,
        config: TemporalEngineConfig | None = None,
        detector_fn: Callable[[Any], list[dict[str, Any]]] | None = None,
        save_visualizations: bool = False,
        output_dir: str | Path | None = None,
    ) -> None:
        self.config = config or TemporalEngineConfig()
        self.engine = TemporalVerificationEngine(config=self.config)
        self.detector_fn = detector_fn
        self.save_visualizations = save_visualizations
        self.output_dir = Path(output_dir) if output_dir else None
        self.frame_processor = FrameProcessor() if save_visualizations else None
        self.frame_records: list[dict[str, Any]] = []

    def process_sequence(
        self,
        frame_sequence: list[tuple[int, Any, list[dict[str, Any]]]],
        fps: float = 30.0,
    ) -> dict[str, Any]:
        """
        Process a sequential stream of frames.
        
        Args:
            frame_sequence: List of tuples: (frame_index, frame_img_or_path, detections).
                If self.detector_fn is provided, detections can be empty and will be inferred.
            fps: Frame rate for timestamp calculations.
            
        Returns:
            Dictionary with complete sequence results, alerts, and summary.
        """
        self.frame_records.clear()
        self.engine.reset()

        for frame_idx, frame_input, raw_dets in frame_sequence:
            # If detector provided, infer detections
            if self.detector_fn is not None and not raw_dets:
                dets = self.detector_fn(frame_input)
            else:
                dets = raw_dets

            # Process through temporal engine
            frame_result = self.engine.process_frame(frame_index=frame_idx, raw_detections=dets, fps=fps)

            # Optionally render and save visualization
            if self.save_visualizations and self.output_dir and self.frame_processor:
                save_path = self.output_dir / f"frame_{frame_idx:06d}.jpg"
                self.frame_processor.annotate_frame(
                    frame_path_or_array=frame_input,
                    frame_data=frame_result,
                    output_path=save_path,
                )

            self.frame_records.append(frame_result)

        summary = self.engine.get_summary()
        summary["total_sequence_frames"] = len(frame_sequence)
        summary["all_alerts"] = [a.to_dict() for a in self.engine.decision_engine.alerts_log]

        return {
            "summary": summary,
            "frame_results": self.frame_records,
            "alerts": summary["all_alerts"],
        }
