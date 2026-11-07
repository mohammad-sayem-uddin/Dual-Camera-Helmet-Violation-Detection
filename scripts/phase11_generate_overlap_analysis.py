#!/usr/bin/env python3
"""
Phase 11.5: Generate PAIR_OVERLAP_ANALYSIS.csv
Classifies all 25 pairs into 4 duration brackets:
- < 2.0s
- 2.0s - 4.0s
- 4.0s - 6.0s
- > 6.0s
Analyzes operational capacity for Phase 10 Temporal Engine (W=10 frames, N_confirm=5 frames = 83.3 ms @ 60fps / 167 ms @ 30fps).
"""
import csv
import json
from pathlib import Path


def main():
    root = Path(".")
    meta_path = root / "results/phase11_dual_camera/synchronization_metadata.json"
    out_csv = root / "results/phase11_dual_camera/human_review/PAIR_OVERLAP_ANALYSIS.csv"
    
    with open(meta_path) as f:
        pairs = json.load(f)
        
    rows = []
    
    # Phase 10 temporal constants
    FPS = 60.0
    BUFFER_W = 10  # frames (166.7 ms @ 60fps)
    N_CONFIRM = 5  # frames (83.3 ms @ 60fps, 166.7 ms @ 30fps)
    TRACK_INIT = 3 # frames (50 ms)
    
    for p in pairs:
        pid = p["pair_id"]
        overlap = p["overlap_duration"]
        frames = int(round(overlap * FPS))
        
        # Duration Bracket
        if overlap < 2.0:
            bracket = "<2s"
            sync_suitability = "SUITABLE_SHORT_PASSBY"
            temp_suitability = "PASS_ADEQUATE_RAPID_PASSBY"
            notes = (
                f"Overlap {overlap:.2f}s ({frames}f) exceeds required 10-frame buffer by {frames/BUFFER_W:.1f}x. "
                f"Adequate for rapid pass-by verification (track init: 3f, buffer: 10f, confirm: 5f)."
            )
        elif 2.0 <= overlap < 4.0:
            bracket = "2-4s"
            sync_suitability = "OPTIMAL_STANDARD_TRANSIT"
            temp_suitability = "PASS_OPTIMAL_TRANSIT_EVALUATION"
            notes = (
                f"Overlap {overlap:.2f}s ({frames}f) provides {frames/BUFFER_W:.1f}x sliding window capacity. "
                f"Permits full track initialization, steady-state classification, and citation latching."
            )
        elif 4.0 <= overlap < 6.0:
            bracket = "4-6s"
            sync_suitability = "OPTIMAL_MULTI_TRAJECTORY"
            temp_suitability = "PASS_EXTENDED_TRACKING_BENCHMARK"
            notes = (
                f"Overlap {overlap:.2f}s ({frames}f) provides {frames/BUFFER_W:.1f}x sliding window capacity. "
                f"Ideal for multi-rider interaction, lane shifts, and continuous confidence accumulation."
            )
        else: # > 6.0s
            bracket = ">6s"
            sync_suitability = "OPTIMAL_LONG_HORIZON"
            temp_suitability = "PASS_LONG_HORIZON_STRESS_BENCHMARK"
            notes = (
                f"Overlap {overlap:.2f}s ({frames}f) provides {frames/BUFFER_W:.1f}x sliding window capacity. "
                f"Benchmark for track recovery across brief visual occlusions and persistent temporal verification."
            )
            
        row = {
            "pair_id": pid,
            "back_video": p["back_video"],
            "front_video": p["front_video"],
            "overlap_duration_sec": f"{overlap:.3f}",
            "overlap_frames_60fps": frames,
            "duration_bracket": bracket,
            "window_capacity_factor": f"{frames / BUFFER_W:.1f}x",
            "n_confirm_capacity_factor": f"{frames / N_CONFIRM:.1f}x",
            "synchronization_testing_suitability": sync_suitability,
            "temporal_violation_evaluation_suitability": temp_suitability,
            "operational_notes": notes
        }
        rows.append(row)
        
    fieldnames = [
        "pair_id", "back_video", "front_video", "overlap_duration_sec", "overlap_frames_60fps",
        "duration_bracket", "window_capacity_factor", "n_confirm_capacity_factor",
        "synchronization_testing_suitability", "temporal_violation_evaluation_suitability",
        "operational_notes"
    ]
    
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Generated {out_csv} with {len(rows)} entries.")


if __name__ == "__main__":
    main()
