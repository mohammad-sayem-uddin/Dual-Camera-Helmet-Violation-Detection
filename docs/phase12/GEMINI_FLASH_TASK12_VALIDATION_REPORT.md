# GEMINI FLASH TASK-12 VALIDATION REPORT — Independent Read-Only Audit of Blind Human GT `v1.1` Packaging & Stage K `tiou()` Fix (Gate D)

**Date:** 2026-11-12  
**Role:** Independent Engineering, Reproducibility & Anti-Fabrication Auditor (`research` read-only subagent)  
**Audited Commit:** `e38bb17ec0c0905c486d91377501d393187b6c6d`  
**Implementation Report Audited:** [`SONNET_TASK12_IMPLEMENTATION_REPORT.md`](file:///Users/sayemuddin/Desktop/research_helmet/docs/phase12/SONNET_TASK12_IMPLEMENTATION_REPORT.md)  
**Protocol Specification:** [`ARGON_TASK12_BLIND_GT_V1_1_PROTOCOL.md`](file:///Users/sayemuddin/Desktop/research_helmet/docs/phase12/ARGON_TASK12_BLIND_GT_V1_1_PROTOCOL.md)  

**Verdict: PASS WITH LIMITATIONS**

All engineering, deterministic dual-replay, offline unit/integration test (`198/198`), Stage J legacy event parity (`85/85`), frozen SHA-256 checksum (`39/39 + 43/43 + 381/381 + 8/8`), and anti-fabrication checks **PASS** with zero workspace mutations. Two non-blocking limitations/nuances are documented in §6 (including the intentional `PENDING_HUMAN_REVIEW` status of `v1.1`).

---

## 1. Git State & Cleanliness Audit

- **`git rev-parse HEAD`:** `e38bb17ec0c0905c486d91377501d393187b6c6d` (`TASK-12: blind human GT v1.1 workqueue package and Stage K point-interval tIoU fix`, `20 files changed, +7226 / -5`).
- **`git status --short` before & after read-only audit:** Only pre-existing untracked `?? docs/phase12/GEMINI_FLASH_TASK09_VALIDATION_REPORT.md`; zero files created or modified inside the repository during validation (`.pytest_cache` timestamp unchanged).

---

## 2. Code & Stage K `tiou()` Fix Audit (`scripts/reproducibility/evaluate.py`)

1. **`compute_tiou(a, b, *, mode='fixed', tol=1e-9)` (`evaluate.py` L33–43):**
   - Verified matching 1-frame `[t, t]` intervals (`(1.428344, 1.428344)`, `(-1.420111, -1.420111)`, `(0.0, 0.0)`): `fixed = 1.0`, `legacy = 0.0`.
   - Verified disjoint point intervals (`(1.0, 1.0)` vs `(2.0, 2.0)`): `0.0` in both modes.
   - Verified point vs non-empty intervals (`(1.0, 1.0)` vs `(1.0, 2.0)`, `(1.5, 1.5)` vs `(1.0, 2.0)`): `0.0` in both modes.
   - Verified missing windows (any `None` bound): `None` in both modes.
   - Verified non-degenerate intervals (`(0.0, 2.0)` vs `(1.0, 3.0)` -> `0.3333333333333333`, `(0.0, 1.0)` vs `(2.0, 3.0)` -> `0.0`, `(1.0, 4.0)` vs `(1.0, 4.0)` -> `1.0`): identical across modes.
2. **CLI & `run_evaluation()` wiring (`evaluate.py` L46–54, L443–444, L580–593):**
   - `run_evaluation(..., tiou_mode='fixed')` validates `tiou_mode in ('fixed', 'legacy')`; CLI `--tiou-mode {fixed,legacy}` defaults to `fixed`.
3. **Exact parity & surgical impact scope:**
   - `legacy_eval` matches frozen `results/phase12_integrated/` `stagek_results['events']` exactly (`True`) and produces **byte-for-byte identical** `events_detail.csv` and `sync_by_pair.csv` (`True`), differing at the top level only by expected pre-existing post-freeze GAP-B3/GAP-B4 fields (`helmet_track_level_redesign`, `association_rider_item_level`, `not_computable`).
   - Across all GT and Stage J Pred events, `('pair_012', 'front_moto_0003', '')` `(1.428344, 1.428344)` and `('pair_015', 'front_moto_0001', '')` `(-1.420111, -1.420111)` are the sole `w0 == w1` events; only `events` and `per_pair_stats` for `pair_012` (`TP 7->8, FP 3->2, FN 1->0`) and `pair_015` (`TP 1->2, FP 2->1, FN 2->1`) change, leaving all non-event metrics invariant.

---

## 3. Blind Human GT `v1.1` Anti-Fabrication & Honesty Audit

Audited all files under `data/dual_camera_test_v1/releases/dual_camera_gt_v1.1_blind_human/`:
- `release_metadata.json` validates against `schema.json` (`jsonschema.validate`: PASS).
- **Honesty metadata verified:**
  - `human_verification_status == "PENDING_HUMAN_REVIEW"`
  - `human_verified == False`
  - `executed_human_units == 0`
  - `pending_human_units == 92`
  - `fabricated_labels_present == False`
- **Full-table CSV inspection (`blind_annotation_workqueue.csv`: `92` rows, `31` columns):**
  - Pairs covered: `25/25`; source origins: `70` `PHYSICAL_TRANSIT_CORPUS` + `22` `GAPC_UNLINKED_OR_SPLIT_CANDIDATE`; strata: `STRATUM_1_GAPC_PILLION_OR_CONFLICT: 46`, `STRATUM_2_UNCERTAIN_AMBIGUOUS_OR_DELTA: 33`, `STRATUM_3_BASELINE_AND_PLATE_AUDIT: 13`.
  - All `17` blind response columns (`annotator_a_*` `[7]`, `annotator_b_*` `[7]`, `adjudicator_*` `[3]`) are `""` in `92/92` rows (`0 / 1,564` filled cells).
  - `review_status == "PENDING_HUMAN_REVIEW"` in `92/92` rows.
  - Zero AI verdict labels (`DRIVER_NO_HELMET`, `BOTH_NO_HELMET`, `PILLION_NO_HELMET`, `NO_VIOLATION`, `INSUFFICIENT_VISUAL_EVIDENCE`, `HELMET`, `NO_HELMET`, `occluded_unknown`) appear in any column.

---

## 4. Dual Deterministic Replay Verification

Executed two independent runs into `/tmp/`:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:scripts/reproducibility /opt/homebrew/opt/python@3.11/bin/python3.11 scripts/reproducibility/evaluate_task12_audit.py --out /tmp/task12_verify_run1
# -> ALL_CHECKS_PASSED: True (6/6), WALL_SEC: 32.12, PEAK_RSS_MB: 214.96

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:scripts/reproducibility /opt/homebrew/opt/python@3.11/bin/python3.11 scripts/reproducibility/evaluate_task12_audit.py --out /tmp/task12_verify_run2
# -> ALL_CHECKS_PASSED: True (6/6), WALL_SEC: 33.079, PEAK_RSS_MB: 214.97
```

- **Exact byte-for-byte parity (`run1 == run2 == committed results/phase12_task12/`):** Confirmed across all `9` deterministic files: `task12_tiou_comparison.csv`, `legacy_eval/{events_detail.csv, per_pair_stats.json, stagek_results.json, sync_by_pair.csv}`, and `fixed_eval/{events_detail.csv, per_pair_stats.json, stagek_results.json, sync_by_pair.csv}`.
- **`task12_tiou_comparison.json`:** Exact value parity after excluding volatile `runtime_metrics` (`total_wall_sec`, `peak_rss_mb`).
- **Metrics verified:** Legacy `P=0.7529, R=0.9143, F1=0.8258 (TP=64, FP=21, FN=6)` -> Fixed `P=0.7765, R=0.9429, F1=0.8516 (TP=66, FP=19, FN=4)`; `DRIVER_NO_HELMET` F1 `0.8519 -> 0.9259`; `timing_n` `34 -> 36` (`d_onset_mean_abs=0.0`, `d_end_mean_abs=0.0`).

---

## 5. Offline Tests, Stage J Parity & Frozen Checksums

- **Offline pytest suite (`tests/` excluding `test_phase12_pipeline_real_video.py`):**
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest tests/ -q -p no:cacheprovider --ignore=tests/test_phase12_pipeline_real_video.py` -> **`198 passed, 5 warnings in 48.06s`** (`0` failed).
- **Read-only Stage J parity (`event_regression_task04.py` logic via `python3.11 -c`):** **`events compared=85 changed=0`**.
- **Frozen Checksum Integrity (`hashlib.sha256`):**
  - `data/dual_camera_test_v1/annotations/checksums/annotation_checksums.sha256`: **`39/39 OK`**
  - `data/dual_camera_test_v1/releases/dual_camera_gt_v1.0_ai_assisted_audited/checksums/release_checksums.sha256`: **`43/43 OK`**
  - `PACKAGE_MANIFEST.sha256` (entries matching `results/phase12_stagej/`): **`381/381 OK`**
  - `results/phase12_integrated/checksums.sha256`: **`8/8 OK`**

---

## 6. Disclosed Limitations & Nuances

1. **Human Review Intentionally Pending (`PENDING_HUMAN_REVIEW`):** By design and anti-fabrication policy, `dual_camera_gt_v1.1_blind_human/` contains `0` executed human labels (`executed_human_units: 0`); Pass 1–3 blind human annotation and Cohen's $\kappa$ acceptance gating ($\ge 0.85$ / $\ge 0.90$) remain future human execution work.
2. **Protocol Draft Count Typo at `e38bb17` (Reconciled Post-Audit):** At commit `e38bb17`, [`ARGON_TASK12_BLIND_GT_V1_1_PROTOCOL.md`](file:///Users/sayemuddin/Desktop/research_helmet/docs/phase12/ARGON_TASK12_BLIND_GT_V1_1_PROTOCOL.md) §3 line 62 stated `70 + 16 = 86` items, whereas the finalized workqueue, metadata, README, implementation report, and roadmap all contained `70 + 22 = 92` items (`pair_007: 3, pair_009: 7, pair_011: 3, pair_012: 2, pair_013: 4, pair_014: 1, pair_015: 1, pair_020: 1`). Reconciled cleanly in §3 post-audit without altering any release data.
3. **Minor Code Nuances:** `compute_tiou(..., mode='invalid')` does not itself raise `ValueError` (mode validation is enforced cleanly in `run_evaluation()` and CLI `--tiou-mode`); `scripts/reproducibility/evaluate_gapb_redesign.py:651` calls `run_evaluation(...)` without overriding `tiou_mode`, so re-running that legacy script would use the fixed `TP=66 / F1=0.8516` default unless `--tiou-mode legacy` is passed explicitly.
