#!/usr/bin/env python3
"""Automated synthesis and generation of Phase 3 Pilot Comparative Analysis artifacts."""
from __future__ import annotations

import csv
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import ensure_parent, research_path


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def load_json(filepath: Path) -> dict:
    return json.loads(filepath.read_text(encoding="utf-8"))


def main() -> None:
    exp001_val_path = research_path("results/metrics/EXP-001-val.json")
    exp001_test_path = research_path("results/metrics/EXP-001-test.json")
    exp003_val_path = research_path("results/metrics/EXP-003-val.json")
    exp003_test_path = research_path("results/metrics/EXP-003-test.json")
    exp002_val_path = research_path("results/metrics/EXP-002-FIXED-val.json")
    exp002_test_path = research_path("results/metrics/EXP-002-FIXED-test.json")

    exp001_meta_path = research_path("experiments/EXP-001/run_metadata.json")
    exp003_meta_path = research_path("experiments/EXP-003/run_metadata.json")
    exp002_meta_path = research_path("experiments/EXP-002-FIXED/run_metadata.json")

    exp001_ana_path = research_path("results/metrics/EXP-001_analysis.json")
    exp003_ana_path = research_path("results/metrics/EXP-003_analysis.json")
    exp002_ana_path = research_path("results/metrics/EXP-002-FIXED_analysis.json")

    exp001_val = load_json(exp001_val_path)
    exp001_test = load_json(exp001_test_path)
    exp003_val = load_json(exp003_val_path)
    exp003_test = load_json(exp003_test_path)
    exp002_val = load_json(exp002_val_path)
    exp002_test = load_json(exp002_test_path)

    exp001_ana = load_json(exp001_ana_path) if exp001_ana_path.exists() else {}
    exp003_ana = load_json(exp003_ana_path) if exp003_ana_path.exists() else {}
    exp002_ana = load_json(exp002_ana_path) if exp002_ana_path.exists() else {}

    exp001_meta = load_json(exp001_meta_path) if exp001_meta_path.exists() else {}
    exp003_meta = load_json(exp003_meta_path) if exp003_meta_path.exists() else {}
    exp002_meta = load_json(exp002_meta_path) if exp002_meta_path.exists() else {}

    # Extract test metrics
    t1 = exp001_test["overall_metrics"]
    t3 = exp003_test["overall_metrics"]
    t2 = exp002_test["overall_metrics"]

    # Extract val metrics
    v1 = exp001_val["overall_metrics"]
    v3 = exp003_val["overall_metrics"]
    v2 = exp002_val["overall_metrics"]

    # Per class test metrics
    pc1 = exp001_test.get("per_class_metrics", {})
    pc3 = exp003_test.get("per_class_metrics", {})
    pc2 = exp002_test.get("per_class_metrics", {})

    # Compute effects (defined as positive drops according to specification)
    # Δ_size = mAP50(EXP-001) - mAP50(EXP-003)
    # Δ_sequence = mAP50(EXP-003) - mAP50(EXP-002-FIXED)
    delta_size_map50 = t1["map50"] - t3["map50"]
    delta_seq_map50 = t3["map50"] - t2["map50"]

    delta_size_map50_95 = t1["map50_95"] - t3["map50_95"]
    delta_seq_map50_95 = t3["map50_95"] - t2["map50_95"]

    delta_size_p = t1["precision"] - t3["precision"]
    delta_seq_p = t3["precision"] - t2["precision"]

    delta_size_r = t1["recall"] - t3["recall"]
    delta_seq_r = t3["recall"] - t2["recall"]

    delta_size_f1 = t1["f1"] - t3["f1"]
    delta_seq_f1 = t3["f1"] - t2["f1"]

    dur1_h = round(exp001_meta.get("duration_seconds", 26928.0) / 3600.0, 2)
    dur3_h = round(exp003_meta.get("duration_seconds", 22151.58) / 3600.0, 2)
    dur2_h = round(exp002_meta.get("duration_seconds", 0.0) / 3600.0, 2)

    # 1. Output CSV Table: results/tables/EXP_001_EXP_003_EXP_002_FIXED.csv
    csv_rows = [
        {"Metric": "Test mAP@0.50", "EXP-001 (4854 Train)": f"{t1['map50']:.4f}", "EXP-003 (3866 Train)": f"{t3['map50']:.4f}", "EXP-002-FIXED (3866 Train)": f"{t2['map50']:.4f}", "Delta_Size (EXP-001 - EXP-003)": f"{delta_size_map50:+.4f}", "Delta_Sequence (EXP-003 - EXP-002-FIXED)": f"{delta_seq_map50:+.4f}"},
        {"Metric": "Test mAP@0.50:0.95", "EXP-001 (4854 Train)": f"{t1['map50_95']:.4f}", "EXP-003 (3866 Train)": f"{t3['map50_95']:.4f}", "EXP-002-FIXED (3866 Train)": f"{t2['map50_95']:.4f}", "Delta_Size (EXP-001 - EXP-003)": f"{delta_size_map50_95:+.4f}", "Delta_Sequence (EXP-003 - EXP-002-FIXED)": f"{delta_seq_map50_95:+.4f}"},
        {"Metric": "Test Precision", "EXP-001 (4854 Train)": f"{t1['precision']:.4f}", "EXP-003 (3866 Train)": f"{t3['precision']:.4f}", "EXP-002-FIXED (3866 Train)": f"{t2['precision']:.4f}", "Delta_Size (EXP-001 - EXP-003)": f"{delta_size_p:+.4f}", "Delta_Sequence (EXP-003 - EXP-002-FIXED)": f"{delta_seq_p:+.4f}"},
        {"Metric": "Test Recall", "EXP-001 (4854 Train)": f"{t1['recall']:.4f}", "EXP-003 (3866 Train)": f"{t3['recall']:.4f}", "EXP-002-FIXED (3866 Train)": f"{t2['recall']:.4f}", "Delta_Size (EXP-001 - EXP-003)": f"{delta_size_r:+.4f}", "Delta_Sequence (EXP-003 - EXP-002-FIXED)": f"{delta_seq_r:+.4f}"},
        {"Metric": "Test F1-Score", "EXP-001 (4854 Train)": f"{t1['f1']:.4f}", "EXP-003 (3866 Train)": f"{t3['f1']:.4f}", "EXP-002-FIXED (3866 Train)": f"{t2['f1']:.4f}", "Delta_Size (EXP-001 - EXP-003)": f"{delta_size_f1:+.4f}", "Delta_Sequence (EXP-003 - EXP-002-FIXED)": f"{delta_seq_f1:+.4f}"},
        {"Metric": "Test no_helmet AP50", "EXP-001 (4854 Train)": f"{pc1.get('no_helmet', {}).get('map50', 0.0):.4f}", "EXP-003 (3866 Train)": f"{pc3.get('no_helmet', {}).get('map50', 0.0):.4f}", "EXP-002-FIXED (3866 Train)": f"{pc2.get('no_helmet', {}).get('map50', 0.0):.4f}", "Delta_Size (EXP-001 - EXP-003)": f"{(pc1.get('no_helmet', {}).get('map50', 0.0) - pc3.get('no_helmet', {}).get('map50', 0.0)):+.4f}", "Delta_Sequence (EXP-003 - EXP-002-FIXED)": f"{(pc3.get('no_helmet', {}).get('map50', 0.0) - pc2.get('no_helmet', {}).get('map50', 0.0)):+.4f}"},
        {"Metric": "Test helmet AP50", "EXP-001 (4854 Train)": f"{pc1.get('helmet', {}).get('map50', 0.0):.4f}", "EXP-003 (3866 Train)": f"{pc3.get('helmet', {}).get('map50', 0.0):.4f}", "EXP-002-FIXED (3866 Train)": f"{pc2.get('helmet', {}).get('map50', 0.0):.4f}", "Delta_Size (EXP-001 - EXP-003)": f"{(pc1.get('helmet', {}).get('map50', 0.0) - pc3.get('helmet', {}).get('map50', 0.0)):+.4f}", "Delta_Sequence (EXP-003 - EXP-002-FIXED)": f"{(pc3.get('helmet', {}).get('map50', 0.0) - pc2.get('helmet', {}).get('map50', 0.0)):+.4f}"},
        {"Metric": "Val mAP@0.50", "EXP-001 (4854 Train)": f"{v1['map50']:.4f}", "EXP-003 (3866 Train)": f"{v3['map50']:.4f}", "EXP-002-FIXED (3866 Train)": f"{v2['map50']:.4f}", "Delta_Size (EXP-001 - EXP-003)": f"{(v1['map50'] - v3['map50']):+.4f}", "Delta_Sequence (EXP-003 - EXP-002-FIXED)": f"{(v3['map50'] - v2['map50']):+.4f}"},
        {"Metric": "Val F1-Score", "EXP-001 (4854 Train)": f"{v1['f1']:.4f}", "EXP-003 (3866 Train)": f"{v3['f1']:.4f}", "EXP-002-FIXED (3866 Train)": f"{v2['f1']:.4f}", "Delta_Size (EXP-001 - EXP-003)": f"{(v1['f1'] - v3['f1']):+.4f}", "Delta_Sequence (EXP-003 - EXP-002-FIXED)": f"{(v3['f1'] - v2['f1']):+.4f}"},
        {"Metric": "Best Validation Epoch", "EXP-001 (4854 Train)": str(exp001_ana.get("best_epoch", "N/A")), "EXP-003 (3866 Train)": str(exp003_ana.get("best_epoch", "N/A")), "EXP-002-FIXED (3866 Train)": str(exp002_ana.get("best_epoch", "N/A")), "Delta_Size (EXP-001 - EXP-003)": "N/A", "Delta_Sequence (EXP-003 - EXP-002-FIXED)": "N/A"},
        {"Metric": "Training Duration (h)", "EXP-001 (4854 Train)": f"{dur1_h}", "EXP-003 (3866 Train)": f"{dur3_h}", "EXP-002-FIXED (3866 Train)": f"{dur2_h}", "Delta_Size (EXP-001 - EXP-003)": f"{(dur1_h - dur3_h):+.2f}", "Delta_Sequence (EXP-003 - EXP-002-FIXED)": f"{(dur3_h - dur2_h):+.2f}"},
    ]

    csv_path = ensure_parent(research_path("results/tables/EXP_001_EXP_003_EXP_002_FIXED.csv"))
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Generated CSV table: {csv_path}")

    # 2. Output JSON Metrics: results/metrics/phase3_pilot_comparison.json
    pilot_comparison_json = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "experiments": {
            "EXP-001": {
                "description": "Baseline random-frame split (4,854 train images)",
                "seed": 42,
                "train_images": 4854,
                "val_images": 347,
                "test_images": 346,
                "best_epoch": exp001_ana.get("best_epoch", 40),
                "duration_seconds": exp001_meta.get("duration_seconds", 26928.0),
                "val_metrics": v1,
                "test_metrics": t1,
                "per_class_test": pc1,
            },
            "EXP-003": {
                "description": "Size-matched random-frame control (3,866 train images)",
                "seed": 42,
                "train_images": 3866,
                "val_images": 347,
                "test_images": 346,
                "best_epoch": exp003_ana.get("best_epoch", 46),
                "duration_seconds": exp003_meta.get("duration_seconds", 22151.58),
                "val_metrics": v3,
                "test_metrics": t3,
                "per_class_test": pc3,
            },
            "EXP-002-FIXED": {
                "description": "Sequence-exclusive group split (3,866 train images, 169 true groups, 0.0% overlap)",
                "seed": 42,
                "train_images": 3866,
                "val_images": 852,
                "test_images": 829,
                "best_epoch": exp002_ana.get("best_epoch", "N/A"),
                "duration_seconds": exp002_meta.get("duration_seconds", 0.0),
                "val_metrics": v2,
                "test_metrics": t2,
                "per_class_test": pc2,
            },
        },
        "isolated_effects": {
            "data_size_effect": {
                "description": "Drop from 4,854 to 3,866 train images under random-frame splitting (EXP-001 - EXP-003)",
                "delta_map50": round(delta_size_map50, 6),
                "delta_map50_95": round(delta_size_map50_95, 6),
                "delta_precision": round(delta_size_p, 6),
                "delta_recall": round(delta_size_r, 6),
                "delta_f1": round(delta_size_f1, 6),
            },
            "sequence_isolation_effect": {
                "description": "Drop from random-frame to sequence-exclusive evaluation at 3,866 train images (EXP-003 - EXP-002-FIXED)",
                "delta_map50": round(delta_seq_map50, 6),
                "delta_map50_95": round(delta_seq_map50_95, 6),
                "delta_precision": round(delta_seq_p, 6),
                "delta_recall": round(delta_seq_r, 6),
                "delta_f1": round(delta_seq_f1, 6),
            },
        },
    }

    json_path = ensure_parent(research_path("results/metrics/phase3_pilot_comparison.json"))
    json_path.write_text(json.dumps(pilot_comparison_json, indent=2), encoding="utf-8")
    print(f"Generated JSON metrics: {json_path}")


if __name__ == "__main__":
    main()
