# SONNET TASK-12 IMPLEMENTATION REPORT — Blind Human GT `v1.1` Packaging (`PENDING_HUMAN_REVIEW`) & Stage K `tiou()` Fix

**Date:** 2026-11-12  
**Role:** Senior Computer Vision & Reproducibility Engineer  
**Design Authority:** [`ARGON_TASK12_BLIND_GT_V1_1_PROTOCOL.md`](file:///Users/sayemuddin/Desktop/research_helmet/docs/phase12/ARGON_TASK12_BLIND_GT_V1_1_PROTOCOL.md)  
**Prerequisite Cleared Gates:** `TASK-09B` (`8cc3c96`), `TASK-10` (`00bd15f`), `TASK-11` (`7763cd3`)

---

## 1. Executive Summary

Implemented `TASK-12` across two deliverables cleanly decoupled from frozen `v1.0` and Stage K artifacts:

1. **Non-Destructive Blind Human Ground Truth `v1.1` Protocol & Workqueue Package (`PENDING_HUMAN_REVIEW`):**
   - Authored [`ARGON_TASK12_BLIND_GT_V1_1_PROTOCOL.md`](file:///Users/sayemuddin/Desktop/research_helmet/docs/phase12/ARGON_TASK12_BLIND_GT_V1_1_PROTOCOL.md) defining blind two-annotator + independent adjudicator verification, unweighted Cohen's $\kappa$ acceptance thresholds ($\kappa \ge 0.85$ binary violation / rider helmet, $\kappa \ge 0.90$ cross-camera link), and rear license-plate readability tiers (`PLATE_CLEAR`, `PLATE_PARTIAL`, `PLATE_UNREADABLE_MOTION_BLUR`, `PLATE_NOT_VISIBLE_FRONT_ONLY`).
   - Built isolated release package `data/dual_camera_test_v1/releases/dual_camera_gt_v1.1_blind_human/` (`README.md`, `release_metadata.json`, `schema.json`, `blind_annotation_workqueue.csv` containing `92` stratified items across all `25` pairs: `46` Stratum 1 GAP-C pillion/conflict divergence, `33` Stratum 2 uncertain/ambiguous/delta, `13` Stratum 3 baseline & rear plate audit).
   - **Zero Fabricated Labels:** `release_metadata.json` explicitly declares `"human_verification_status": "PENDING_HUMAN_REVIEW"`, `"human_verified": false`, `"executed_human_units": 0`, `"fabricated_labels_present": false`, and all `17` blind annotator/adjudicator columns across all `92` CSV rows contain zero filled verdicts. `dual_camera_gt_v1.0_ai_assisted_audited/` remains 100% untouched.

2. **Point-Interval `tiou()` Bug Fix & Non-Destructive Side-by-Side Comparison (`results/phase12_task12/`):**
   - Extracted [`compute_tiou()`](file:///Users/sayemuddin/Desktop/research_helmet/scripts/reproducibility/evaluate.py#L34-L44) inside `scripts/reproducibility/evaluate.py` (`--tiou-mode {fixed,legacy}`, defaulting to `fixed`) so degenerate matching 1-frame intervals (`[t, t]` vs `[t, t]`, `union <= 1e-9`) score `1.0` rather than `0.0`, while disjoint points/spans continue returning `0.0`.
   - Executed non-destructive side-by-side Stage K evaluation into `results/phase12_task12/` (`scripts/reproducibility/evaluate_task12_audit.py`), leaving frozen `results/phase12_integrated/` completely untouched.
   - Verified **`6/6` automated checks (`ALL_CHECKS_PASSED: True`)**, **`198 passed`** in offline pytest (`tests/test_phase12_task12_eval_fix.py` adding 3 tests), and **`events compared=85 changed=0`** in `event_regression_task04.py`.

---

## 2. Side-by-Side Stage K Evaluation Comparison (`results/phase12_task12/task12_tiou_comparison.json`)

| Metric | Legacy `tiou` (`legacy_eval`) | Fixed `tiou` (`fixed_eval`) | Delta (`Fixed - Legacy`) |
|---|---:|---:|---:|
| Overall Event Precision | 0.7529 | **0.7765** | +0.0236 |
| Overall Event Recall | 0.9143 | **0.9429** | +0.0286 |
| Overall Event F1 | 0.8258 | **0.8516** | **+0.0258** |
| True Positives (`tp`) | 64 | **66** | **+2** |
| False Positives (`fp`) | 21 | **19** | **-2** |
| False Negatives (`fn`) | 6 | **4** | **-2** |
| Strict Identity Matches | 65 | **67** | +2 |
| `DRIVER_NO_HELMET` F1 (`tp/fp/fn`) | 0.8519 (`23/4/4`) | **0.9259 (`25/2/2`)** | **+0.0740** (`+2 TP, -2 FP, -2 FN`) |
| Timed Match Count (`timing_n`) | 34 | **36** | +2 (`d_onset_mean_abs=0.0`, `d_end_mean_abs=0.0`) |
| Unmatched `would_match_differs_only_by_excluded_track` (Pred / GT) | 2 / 5 | **0 / 3** | -2 / -2 |

**Exact Recovered Matches:**
1. `pair_012 / front_moto_0003`: `DRIVER_NO_HELMET`, `w=[1.428344, 1.428344]`, `tiou=1.0`, `d_onset=0.0`, `d_end=0.0`.
2. `pair_015 / front_moto_0001`: `DRIVER_NO_HELMET`, `w=[-1.420111, -1.420111]`, `tiou=1.0`, `d_onset=0.0`, `d_end=0.0`.

**Non-Event Metrics Invariant:** `detection_head_operating_point`, `association_motorcycle`, and `association_rider_item_level` match identically across `legacy_eval` and `fixed_eval`; `legacy_eval` matches frozen `results/phase12_integrated/` `events`, `events_detail.csv`, and `sync_by_pair.csv` **100% exactly**.

---

## 3. Verification Commands & Observed Results

```bash
PYTHONPATH=.:scripts/reproducibility /opt/homebrew/opt/python@3.11/bin/python3.11 scripts/reproducibility/evaluate_task12_audit.py --out results/phase12_task12
# -> TASK12_OUT_DIR: results/phase12_task12
# -> ALL_CHECKS_PASSED: True (6/6)
# -> WALL_SEC: 31.743 PEAK_RSS_MB: 214.97

PYTHONPATH=. /opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest tests/ -q --ignore=tests/test_phase12_pipeline_real_video.py
# -> 198 passed, 5 warnings in 45.36s

PYTHONPATH=. /opt/homebrew/opt/python@3.11/bin/python3.11 scripts/reproducibility/event_regression_task04.py
# -> events compared=85 changed=0
```
