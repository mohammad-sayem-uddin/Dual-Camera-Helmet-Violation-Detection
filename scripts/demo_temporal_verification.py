#!/usr/bin/env python3
"""
Command-Line Demonstration of Confidence-Aware Temporal Verification Framework.

Simulates roadside traffic surveillance video streams and demonstrates:
- Detection ingestion and IoU tracking
- Temporal state evolution across the 6-state FSM
- Multi-frame confidence aggregation (Voting, Confidence-Weighted, EMA)
- Hysteresis violation decision gating and alert emission
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.event_evaluator import generate_synthetic_scenarios, run_full_synthetic_benchmark
from src.temporal.temporal_engine import TemporalEngineConfig, TemporalVerificationEngine


def run_scenario_demo(scenario_key: str, aggregator: str = "confidence_weighted") -> None:
    scenarios = generate_synthetic_scenarios()
    if scenario_key not in scenarios:
        print(f"Error: Unknown scenario '{scenario_key}'.")
        print(f"Available scenarios: {list(scenarios.keys())}")
        sys.exit(1)

    s_data = scenarios[scenario_key]
    print("=" * 80)
    print(f"TEMPORAL VERIFICATION DEMONSTRATION: {scenario_key.upper()}")
    print(f"Description: {s_data['description']}")
    print(f"Aggregator:  {aggregator}")
    print("=" * 80)

    config = TemporalEngineConfig(aggregator_method=aggregator, sliding_window_size=10)
    engine = TemporalVerificationEngine(config=config)

    frames = s_data["frames"]
    fps = 30.0

    print(f"{'FRAME':<7} | {'TIME':<7} | {'TRACKS':<7} | {'ACTIVE TRACK DETAILS':<42} | {'ALERTS'}")
    print("-" * 80)

    for f_idx, raw_dets in enumerate(frames):
        res = engine.process_frame(frame_index=f_idx, raw_detections=raw_dets, fps=fps)
        t_sec = res["timestamp_seconds"]
        n_tracks = res["active_tracks_count"]
        n_alerts = res["new_alerts_count"]

        track_info_parts = []
        for t in res["active_tracks"]:
            tid = t["track_id"]
            state = t["state"]
            conf = t["aggregated_confidence"]
            p_no_h = conf.get("no_helmet", 0.0)
            p_h = conf.get("helmet", 0.0)
            track_info_parts.append(f"T{tid}:[{state} no_h={p_no_h:.2f} h={p_h:.2f}]")

        track_str = ", ".join(track_info_parts) if track_info_parts else "(No detections / occluded)"
        alert_str = f"*** VIOLATION ALERT ({n_alerts}) ***" if n_alerts > 0 else "None"

        print(f"{f_idx:<7} | {t_sec:5.2f}s  | {n_tracks:<7} | {track_str:<42} | {alert_str}")

    print("=" * 80)
    print("EXECUTIVE SUMMARY:")
    summary = engine.get_summary()
    print(f"Total Frames Processed:   {summary['total_frames_processed']}")
    print(f"Total Tracks Instantiated:{summary['total_tracks_created']}")
    print(f"Total Violations Emitted: {summary['total_alerts_emitted']}")
    print(f"Final State Distribution: {summary['state_distribution']}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate Confidence-Aware Temporal Verification Pipeline.")
    parser.add_argument("--scenario", default="stable_violation", help="Scenario name (or 'all' to run benchmark)")
    parser.add_argument("--aggregator", default="confidence_weighted", choices=["simple_voting", "confidence_weighted", "ema"])
    parser.add_argument("--export-benchmark", action="store_true", help="Run all scenarios across all aggregators and export results")
    args = parser.parse_args()

    if args.export_benchmark or args.scenario == "all":
        print("Running full synthetic benchmark across all scenarios and aggregators...")
        results = run_full_synthetic_benchmark()
        out_path = Path("results/phase10_temporal/synthetic_benchmark_results.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"Benchmark results successfully exported to {out_path}!")
    else:
        run_scenario_demo(scenario_key=args.scenario, aggregator=args.aggregator)


if __name__ == "__main__":
    main()
