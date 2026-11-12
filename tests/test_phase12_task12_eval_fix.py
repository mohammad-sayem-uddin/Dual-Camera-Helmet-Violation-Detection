"""Unit & integration tests for TASK-12 tIoU fix and GT v1.1 package integrity."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest

import scripts.reproducibility.evaluate as stagek_eval


REPO_ROOT = Path(__file__).resolve().parents[1]
GT_V1_1_DIR = REPO_ROOT / "data" / "dual_camera_test_v1" / "releases" / "dual_camera_gt_v1.1_blind_human"


def test_compute_tiou_handles_point_intervals_without_penalizing_exact_matches() -> None:
    assert stagek_eval.compute_tiou((1.428344, 1.428344), (1.428344, 1.428344), mode="fixed") == 1.0
    assert stagek_eval.compute_tiou((-1.420111, -1.420111), (-1.420111, -1.420111), mode="fixed") == 1.0
    assert stagek_eval.compute_tiou((1.428344, 1.428344), (1.428344, 1.428344), mode="legacy") == 0.0
    # Disjoint point intervals or point vs span must still be 0.0
    assert stagek_eval.compute_tiou((1.0, 1.0), (2.0, 2.0), mode="fixed") == 0.0
    assert stagek_eval.compute_tiou((1.0, 1.0), (1.0, 2.0), mode="fixed") == 0.0
    # Missing windows yield None
    assert stagek_eval.compute_tiou((None, None), (1.0, 2.0), mode="fixed") is None
    # Non-degenerate overlapping windows agree identically across modes
    assert stagek_eval.compute_tiou((0.0, 2.0), (1.0, 3.0), mode="fixed") == pytest.approx(1.0 / 3.0)
    assert stagek_eval.compute_tiou((0.0, 2.0), (1.0, 3.0), mode="legacy") == pytest.approx(1.0 / 3.0)


def test_stagek_run_evaluation_recovers_pair012_and_pair015_only() -> None:
    res_leg, _ = stagek_eval.run_evaluation(write_files=False, tiou_mode="legacy")
    res_fix, _ = stagek_eval.run_evaluation(write_files=False, tiou_mode="fixed")

    ev_leg = res_leg["events"]
    ev_fix = res_fix["events"]
    assert ev_leg["tp"] == 64 and ev_leg["fp"] == 21 and ev_leg["fn"] == 6
    assert ev_fix["tp"] == 66 and ev_fix["fp"] == 19 and ev_fix["fn"] == 4
    assert ev_fix["f1"] == 0.8516

    matched_leg = {(m["pair"], m["front"], m["back"]) for m in ev_leg["matches"]}
    matched_fix = {(m["pair"], m["front"], m["back"]) for m in ev_fix["matches"]}
    recovered = sorted(matched_fix - matched_leg)
    assert recovered == [
        ("pair_012", "front_moto_0003", ""),
        ("pair_015", "front_moto_0001", ""),
    ]
    for m in ev_fix["matches"]:
        if (m["pair"], m["front"], m["back"]) in set(recovered):
            assert m["tiou"] == 1.0
            assert m["d_onset"] == 0.0
            assert m["d_end"] == 0.0


def test_gt_v1_1_blind_human_package_is_honest_and_non_fabricated() -> None:
    meta = json.loads((GT_V1_1_DIR / "release_metadata.json").read_text(encoding="utf-8"))
    assert meta["release_id"] == "dual_camera_gt_v1.1_blind_human"
    assert meta["human_verification_status"] == "PENDING_HUMAN_REVIEW"
    assert meta["human_verified"] is False
    assert meta["executed_human_units"] == 0
    assert meta["fabricated_labels_present"] is False

    with open(GT_V1_1_DIR / "blind_annotation_workqueue.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == meta["workqueue_summary"]["total_items"]
    assert len({r["pair_id"] for r in rows}) == 25
    blind_cols = [c for c in rows[0].keys() if c.startswith(("annotator_a_", "annotator_b_", "adjudicator_"))]
    assert len(blind_cols) >= 14
    for r in rows:
        assert r["review_status"] == "PENDING_HUMAN_REVIEW"
        for col in blind_cols:
            assert r[col] == ""
