"""TASK-12 Reproducible Evaluation Audit: Stage K tIoU Fix & GT v1.1 Package Check.

Runs `scripts/reproducibility/evaluate.py` side-by-side in `legacy` and `fixed`
tIoU modes into `--out` (defaulting to `results/phase12_task12`, never mutating
`results/phase12_integrated/` or `data/dual_camera_test_v1/releases/`), verifies
exact point-interval recovery (`pair_012/front_moto_0003`, `pair_015/front_moto_0001`),
and audits `dual_camera_gt_v1.1_blind_human` anti-fabrication invariants.
"""
from __future__ import annotations

import argparse
import csv
import filecmp
import json
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
_REPRO_DIR = REPO_ROOT / "scripts" / "reproducibility"
if str(_REPRO_DIR) not in sys.path:
    sys.path.insert(0, str(_REPRO_DIR))

import scripts.reproducibility.evaluate as stagek_eval  # noqa: E402


FROZEN_STAGEK_DIR = REPO_ROOT / "results" / "phase12_integrated"
GT_V1_0_DIR = REPO_ROOT / "data" / "dual_camera_test_v1" / "releases" / "dual_camera_gt_v1.0_ai_assisted_audited"
GT_V1_1_DIR = REPO_ROOT / "data" / "dual_camera_test_v1" / "releases" / "dual_camera_gt_v1.1_blind_human"


def _summarize_eval(events: dict[str, Any]) -> dict[str, Any]:
    return {
        "precision": events["precision"],
        "recall": events["recall"],
        "f1": events["f1"],
        "tp": events["tp"],
        "fp": events["fp"],
        "fn": events["fn"],
        "strict_identity_matches": events["strict_identity_matches"],
        "type_agreeing": events["type_agreeing"],
        "driver_no_helmet": events["per_type"]["DRIVER_NO_HELMET"],
        "timing_n": events["timing_errors_sec"]["n"],
        "timing_d_onset_mean_abs": events["timing_errors_sec"]["d_onset_mean_abs"],
        "timing_d_end_mean_abs": events["timing_errors_sec"]["d_end_mean_abs"],
        "unmatched_pred_classes": events["unmatched_pred_classes"],
        "unmatched_gt_classes": events["unmatched_gt_classes"],
    }


def _audit_gt_v1_1_package() -> dict[str, Any]:
    meta_path = GT_V1_1_DIR / "release_metadata.json"
    schema_path = GT_V1_1_DIR / "schema.json"
    wq_path = GT_V1_1_DIR / "blind_annotation_workqueue.csv"
    readme_path = GT_V1_1_DIR / "README.md"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    with open(wq_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    blind_response_cols = [
        c
        for c in (rows[0].keys() if rows else [])
        if c.startswith(("annotator_a_", "annotator_b_", "adjudicator_"))
    ]
    non_empty_blind_cells = sum(
        1 for r in rows for c in blind_response_cols if str(r.get(c, "")).strip() != ""
    )
    all_pending = all(r.get("review_status") == "PENDING_HUMAN_REVIEW" for r in rows)

    return {
        "files_present": all(p.exists() for p in (meta_path, schema_path, wq_path, readme_path)),
        "schema_id": schema.get("$id"),
        "human_verification_status": meta.get("human_verification_status"),
        "human_verified": meta.get("human_verified"),
        "executed_human_units": meta.get("executed_human_units"),
        "pending_human_units": meta.get("pending_human_units"),
        "fabricated_labels_present": meta.get("fabricated_labels_present"),
        "workqueue_rows": len(rows),
        "pairs_covered": len({r["pair_id"] for r in rows}),
        "blind_response_columns_checked": len(blind_response_cols),
        "non_empty_blind_cells": non_empty_blind_cells,
        "all_rows_pending_status": all_pending,
        "v1_0_dir_exists": GT_V1_0_DIR.exists(),
    }


def run_all(out_dir: Path, *, write_files: bool = True) -> dict[str, Any]:
    t0 = time.perf_counter()
    tracemalloc.start()

    legacy_out = out_dir / "legacy_eval"
    fixed_out = out_dir / "fixed_eval"
    r_leg, _ = stagek_eval.run_evaluation(
        out_dir=str(legacy_out) if write_files else None,
        write_files=write_files,
        tiou_mode="legacy",
    )
    r_fix, _ = stagek_eval.run_evaluation(
        out_dir=str(fixed_out) if write_files else None,
        write_files=write_files,
        tiou_mode="fixed",
    )

    frozen = json.loads((FROZEN_STAGEK_DIR / "stagek_results.json").read_text(encoding="utf-8"))
    frozen_events_exact_parity = r_leg["events"] == frozen["events"]

    csv_parity = {}
    if write_files:
        for fname in ("sync_by_pair.csv", "events_detail.csv"):
            csv_parity[fname] = filecmp.cmp(
                FROZEN_STAGEK_DIR / fname,
                legacy_out / fname,
                shallow=False,
            )

    m_leg = {(m["pair"], m["front"], m["back"]): m for m in r_leg["events"]["matches"]}
    m_fix = {(m["pair"], m["front"], m["back"]): m for m in r_fix["events"]["matches"]}
    recovered_keys = sorted(set(m_fix) - set(m_leg))
    recovered_details = [m_fix[k] for k in recovered_keys]

    leg_sum = _summarize_eval(r_leg["events"])
    fix_sum = _summarize_eval(r_fix["events"])
    v1_1_audit = _audit_gt_v1_1_package()

    checks = {
        "c1_legacy_events_exact_parity_with_frozen_stagek": bool(
            frozen_events_exact_parity and all(csv_parity.values() if csv_parity else [True])
        ),
        "c2_fixed_tiou_gains_exact_two_tp": bool(
            fix_sum["tp"] - leg_sum["tp"] == 2
            and leg_sum["fp"] - fix_sum["fp"] == 2
            and leg_sum["fn"] - fix_sum["fn"] == 2
            and leg_sum["tp"] == 64
            and fix_sum["tp"] == 66
        ),
        "c3_recovered_matches_are_pair012_and_pair015_point_intervals": bool(
            recovered_keys == [("pair_012", "front_moto_0003", ""), ("pair_015", "front_moto_0001", "")]
            and all(
                abs(item["tiou"] - 1.0) <= 1e-12
                and abs(item["d_onset"]) <= 1e-12
                and abs(item["d_end"]) <= 1e-12
                for item in recovered_details
            )
        ),
        "c4_non_event_metrics_invariant_to_tiou_fix": bool(
            r_leg["detection_head_operating_point"] == r_fix["detection_head_operating_point"]
            and r_leg["association_motorcycle"] == r_fix["association_motorcycle"]
            and r_leg["association_rider_item_level"] == r_fix["association_rider_item_level"]
        ),
        "c5_gt_v1_1_non_destructive_and_pending_human_review": bool(
            v1_1_audit["files_present"]
            and v1_1_audit["human_verification_status"] == "PENDING_HUMAN_REVIEW"
            and v1_1_audit["human_verified"] is False
            and v1_1_audit["executed_human_units"] == 0
            and v1_1_audit["fabricated_labels_present"] is False
            and v1_1_audit["non_empty_blind_cells"] == 0
            and v1_1_audit["all_rows_pending_status"] is True
            and v1_1_audit["pairs_covered"] == 25
            and v1_1_audit["workqueue_rows"] >= 70
        ),
        "c6_timing_mean_errors_remain_zero_after_recovery": bool(
            fix_sum["timing_n"] == 36
            and fix_sum["timing_d_onset_mean_abs"] == 0.0
            and fix_sum["timing_d_end_mean_abs"] == 0.0
        ),
    }

    _cur, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    wall_sec = round(time.perf_counter() - t0, 3)

    summary = {
        "schema_version": "task12_eval_and_gt_v1_1_v1",
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "legacy_eval_summary": leg_sum,
        "fixed_eval_summary": fix_sum,
        "deltas_fixed_minus_legacy": {
            "tp": fix_sum["tp"] - leg_sum["tp"],
            "fp": fix_sum["fp"] - leg_sum["fp"],
            "fn": fix_sum["fn"] - leg_sum["fn"],
            "f1": round(fix_sum["f1"] - leg_sum["f1"], 4),
            "precision": round(fix_sum["precision"] - leg_sum["precision"], 4),
            "recall": round(fix_sum["recall"] - leg_sum["recall"], 4),
        },
        "recovered_point_interval_events": recovered_details,
        "frozen_csv_byte_parity_legacy": csv_parity,
        "gt_v1_1_package_audit": v1_1_audit,
        "runtime_metrics": {
            "total_wall_sec": wall_sec,
            "peak_rss_mb": round(peak_bytes / (1024 * 1024), 2),
        },
    }

    if write_files:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "task12_tiou_comparison.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        csv_rows = [
            {"mode": "legacy", **leg_sum["driver_no_helmet"], "overall_precision": leg_sum["precision"], "overall_recall": leg_sum["recall"], "overall_f1": leg_sum["f1"], "overall_tp": leg_sum["tp"], "overall_fp": leg_sum["fp"], "overall_fn": leg_sum["fn"], "timing_n": leg_sum["timing_n"]},
            {"mode": "fixed", **fix_sum["driver_no_helmet"], "overall_precision": fix_sum["precision"], "overall_recall": fix_sum["recall"], "overall_f1": fix_sum["f1"], "overall_tp": fix_sum["tp"], "overall_fp": fix_sum["fp"], "overall_fn": fix_sum["fn"], "timing_n": fix_sum["timing_n"]},
        ]
        with open(out_dir / "task12_tiou_comparison.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="TASK-12 Stage K tIoU fix and GT v1.1 package audit")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "phase12_task12")
    args = ap.parse_args()
    res = run_all(args.out, write_files=True)
    print(f"TASK12_OUT_DIR: {args.out}")
    print(f"ALL_CHECKS_PASSED: {res['all_checks_passed']} ({sum(res['checks'].values())}/{len(res['checks'])})")
    print(f"WALL_SEC: {res['runtime_metrics']['total_wall_sec']} PEAK_RSS_MB: {res['runtime_metrics']['peak_rss_mb']}")
    for k, v in res["checks"].items():
        print(f"  CHECK {k}: {v}")
    leg = res["legacy_eval_summary"]
    fix = res["fixed_eval_summary"]
    print(f"  LEGACY: P={leg['precision']} R={leg['recall']} F1={leg['f1']} (TP={leg['tp']}, FP={leg['fp']}, FN={leg['fn']})")
    print(f"  FIXED : P={fix['precision']} R={fix['recall']} F1={fix['f1']} (TP={fix['tp']}, FP={fix['fp']}, FN={fix['fn']})")
    if not res["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
