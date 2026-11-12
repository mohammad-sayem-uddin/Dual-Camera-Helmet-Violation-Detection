# TASK-06A Implementation Report — Keyframe Retention & Simulator Hardening

**Date:** 2026-11-12  
**Project:** An Integrated Dual-Camera Framework for Context-Aware Motorcycle Helmet Violation Detection with Confidence-Aware Temporal Verification  
**Author:** Claude Sonnet 5.5 (Senior Computer Vision & Implementation Engineer)  
**Starting Baseline Commit:** `1c544f3d468082ea2b882eadbce4755e98bfd1cd` (`main`)  
**Previous Validation Basis:** `docs/phase12/GEMINI_FLASH_TASK06_VALIDATION_REPORT.md` (`PASS WITH LIMITATIONS`)

---

## 1. Executive Summary

TASK-06A resolves both concrete engineering defects identified by Gemini 3.8 Flash High during independent TASK-07 validation of TASK-06 (`GAP-B1`):

1. **Issue A (Production Shared-Anchor Eviction & Stale Index Bookkeeping):** Hardened `should_retain_keyframe()` in `src/pipeline/dual_camera_pipeline.py` with explicit global sole-anchor protection and introduced the shared `apply_keyframe_retention_step()` helper used identically by production (`run_camera()`) and offline replay (`scripts/reproducibility/evaluate_gapb_redesign.py`). Any frame serving as the sole retained keyframe for any active or historical track in `track_to_keyframes` is strictly protected against eviction, evicted frame indices are pruned from both `frames_by_index` and every entry of `track_to_keyframes`, and impossible all-sole-anchor capacity conflicts cleanly refuse eviction without dropping protected anchors or exceeding `keyframe_max_per_camera = 24`.
2. **Issue B (Offline Simulator Bookkeeping Divergence & Budget Exceedance):** Rewrote `_replay_camera()` in `scripts/reproducibility/evaluate_gapb_redesign.py` to delegate keyframe retention/eviction directly to `apply_keyframe_retention_step()`, derive `gapb_stored = sorted(sim_keyframes.keys())` directly from the authoritative keyframe store, prune evicted frames from `track_to_kf`, and align `has_det = bool(bundle.motorcycles or bundle.all_heads)` with production Stage G (`dual_camera_pipeline.py`) and Stage J (`annotate_pass1.py`).

Across offline replay of all 25 video pairs (`50` camera streams, `100` motorcycle tracklets, `86` rider subtracks, `130` orphan tracklets):
- Maximum keyframes retained per camera dropped from **`37` (`> 24` budget bug)** to **`24` (`50/50` streams strictly `<= 24`)**.
- Stale `track_to_keyframes` references dropped to **`0`** and dictionary/index sync holds on **`50/50` streams**.
- GAP-B1 motorcycle keyframe coverage and descriptor availability rose from **`98/100`** (`pair_012/front_moto_0007` frame `152` and `pair_012/front_moto_0008` frame `159` lost) to **`100/100`** (`gapb_sim_unavailable_reasons = {}`), with **`0` capacity conflicts**, **`86/86` rider subtracks**, and **`130/130` orphan tracklets** covered in production metadata replay.
- Offline evaluation is bit-deterministic across repeated runs (`diff -ru` empty across all JSON and CSV outputs), while full test verification passes **`181 passed, 0 failed`**.

---

## 2. Root-Cause Verification & Prior Audit Reconciliation

### 2.1 Issue A — Shared-Anchor Eviction in `should_retain_keyframe()` (`dual_camera_pipeline.py`)
In TASK-06 (`1c544f3d`), `should_retain_keyframe()` chose the track `best_tid` with the largest number of stored keyframes (`> 1`) and unconditionally returned `best_frames[-2]` without verifying whether any co-occurring track relied on `best_frames[-2]` as its **only** retained keyframe. In `pair_012/front`:
- Frame `152` captured co-occurring `front_moto_0002`, `front_moto_0004`, `front_moto_0004_rider_s0`, and 1-frame `front_moto_0007`.
- Frame `159` captured co-occurring `front_moto_0002`, `front_moto_0004`, `front_moto_0004_rider_s0`, and 1-frame `front_moto_0008`.
- When later mandatory anchors triggered eviction at frames `171` (`front_moto_0004_rider_s1`) and `193` (`front_moto_0009`), `best_frames[-2]` evicted `152` and then `159`, stripping `front_moto_0007` and `front_moto_0008` of their only keyframes.
- In addition, `run_camera()` previously popped `evict_kf` from `frames_by_index` without removing `evict_kf` from `track_to_keyframes`, leaving stale frame indices inside per-track lists.

### 2.2 Issue B — Unpruned Bookkeeping & Budget Overrun in `_replay_camera()` (`evaluate_gapb_redesign.py`)
In TASK-06, `_replay_camera()` popped `evicted_idx` from `sim_keyframes`, but never removed `evicted_idx` from `gapb_stored` or `track_to_kf`. Consequently:
- `should_retain_keyframe(..., stored_indices=gapb_stored, ...)` continued to see already-popped frames and repeatedly re-evicted stale indices rather than live keys in `sim_keyframes`.
- `sim_keyframes` grew up to `37` frames (violating `KEYFRAME_MAX_PER_CAMERA = 24` on `39/50` streams as audited in `GEMINI_FLASH_TASK06_VALIDATION_REPORT.md`).
- `_replay_camera()` also omitted the `elif frame_index in frames_by_index:` reference update present in `run_camera()`, and set `has_det = bool(bundle.motorcycles or bundle.persons or bundle.all_heads)` instead of `bool(bundle.motorcycles or bundle.all_heads)` used by Pass 1 (`annotate_pass1.py:96`) and production Stage G (`dual_camera_pipeline.py:781`).

---

## 3. Architecture & Algorithm Design

### 3.1 Sole-Anchor Protection & Safe Candidate Selection
In `src/pipeline/dual_camera_pipeline.py`:
1. **`_active_keyframes_by_track()`** normalizes `track_to_keyframes` across both `set[int]` and `list[int]` containers, deduplicating frame indices intersected with `stored_set` and iterating tracks in sorted ID order.
2. **Idempotent stored-frame guard:** If `frame_index in stored_set`, `should_retain_keyframe()` immediately returns `(False, None)` so an already-stored frame never triggers spurious eviction.
3. **Global sole-anchor protection:** Over all tracks in `active_by_tid`, any frame belonging to a single-keyframe track (`len(active_f) == 1`) is added to `protected`.
4. **Candidate filtering:**
   - First, uncoupled frames (`stored_set - referenced_frames`) are evicted in ascending frame order.
   - Next, `safe_candidates = sorted(stored_set - protected)`. Because every frame in `safe_candidates` is absent from `protected`, evicting any single candidate `c` reduces `len(active_by_tid[tid])` from `>= 2` to `>= 1` for every referencing track `tid`, mathematically guaranteeing zero tracks drop to `0` keyframes.
5. **Deterministic multi-criteria eviction ranking (`_eviction_priority_key()`):** Among `safe_candidates`, frames are ranked by:
   - Strictly interior to **all** referencing tracks (`not is_endpoint_anywhere`), preserving entry and exit endpoints for temporal/geometry descriptors.
   - Largest minimum remaining keyframe count after eviction across all referencing tracks (`min_remaining_after_evict`), sacrificing redundant frames from well-sampled tracks rather than shrinking sparsely sampled shared tracks.
   - Largest maximum keyframe count among referencing tracks (`max_track_count`).
   - Fewest referencing tracks (`-len(r_tids)`), minimizing shared blast radius.
   - Largest frame index (`frame_idx`), matching legacy second-most-recent interior preference on single-track ties.
6. **Impossible capacity conflict handling:** If `safe_candidates` is empty (every stored slot is a sole anchor of some track), `should_retain_keyframe()` returns `(False, None)` rather than evicting any protected anchor or exceeding `keyframe_max_per_camera`.

### 3.2 Shared Production & Simulator Helper (`apply_keyframe_retention_step()`)
Both production (`run_camera()` in `dual_camera_pipeline.py`) and offline replay (`_replay_camera()` in `evaluate_gapb_redesign.py`) now execute the exact same state transition via `apply_keyframe_retention_step(...)`:
- Returns `KeyframeRetentionStepResult(retained, evicted_frame_index, needs_anchor, capacity_conflict, observed_track_ids)`.
- Atomically pops `evict_kf` from `frames_by_index` and prunes `evict_kf` from every entry in `track_to_keyframes` (deleting empty track entries via `_prune_evicted_frame()`).
- Enforces an explicit defensive invariant check (`ERR-INTERNAL` if `len(frames_by_index) >= keyframe_max_per_camera` before inserting without eviction).
- Records `frame_index` in `track_to_keyframes` on both newly retained frames and already-stored frames (`elif frame_index in frames_by_index:`).
- In `run_camera()`, `track_to_keyframes` uses `dict[str, set[int]]`, and any `capacity_conflict` logs a clear diagnostic via `_log(...)`.

---

## 4. Modified Files & Symbol Summary

| File | Summary of Changes |
|---|---|
| `src/pipeline/dual_camera_pipeline.py` | Added `KeyframeRetentionStepResult`, `_active_keyframes_by_track()`, `_eviction_priority_key()`, `_prune_evicted_frame()`, `_record_frame_observations()`, `_resize_keyframe()`, `apply_keyframe_retention_step()`. Hardened `should_retain_keyframe()` with sole-anchor protection, sorted deduplication, ranked safe eviction, already-stored guard, and explicit capacity-conflict refusal. Updated `run_camera()` to use `dict[str, set[int]]` and `apply_keyframe_retention_step()`. |
| `scripts/reproducibility/evaluate_gapb_redesign.py` | Updated `_replay_camera()` to use `apply_keyframe_retention_step()`, align `has_det = bool(bundle.motorcycles or bundle.all_heads)`, enforce in-loop cap and subset assertions, remove stale mutable `gapb_stored` drift (`gapb_stored = sorted(sim_keyframes.keys())`), and emit additive audit counters (`gapb_max_keyframes_per_camera_observed`, `gapb_streams_within_keyframe_cap`, `gapb_streams_synced_index_dict`, `gapb_stale_track_keyframe_refs`) in `gapb_evaluation_results.json`. |
| `tests/test_phase12_task06_gapb.py` | Added 10 comprehensive regression tests (`test_task06a_*`) covering shared sole-anchor protection, safe shared eviction, tentative + long-lived co-occurrence, 24-frame saturation, impossible conflict refusal, dict/set insertion-order determinism, list/set pruning, idempotent frame visits, explicit `pair_012/front` frames `152`/`159`/`171`/`193` reproduction, `_replay_camera()` determinism/sync invariants, and 50-stream Stage J metadata replay verification. |
| `docs/phase12/THREE_MODEL_EXECUTION_ROADMAP.md` | Appended TASK-06A status tracking and Section 9 follow-up while preserving historical TASK-06 and TASK-07 records intact. |
| `docs/phase12/SONNET_TASK06A_IMPLEMENTATION_REPORT.md` | Created this implementation and verification report. |

---

## 5. Offline Re-Evaluation Before/After Comparison

Historical TASK-06 committed artifacts in `results/phase12_gapb_redesign/` were preserved untouched. Corrected offline evaluation was executed twice into temporary directories (`GAPB_EVAL_OUT_DIR=/tmp/task06a_gapb_verify1` and `/tmp/task06a_gapb_verify2`), verified byte-identical across all generated JSON/CSV files (`diff -ru` clean), and compared below:

| Metric (`gap_b1_appearance_and_keyframes`) | TASK-06 Committed Baseline | TASK-06A Corrected Replay |
|---|---:|---:|
| `stagej_moto_descriptor_available` | 7 / 100 | 7 / 100 |
| `stagej_unavailable_reasons` | `{"EMPTY_CROP": 56, "NO_MATCHING_FRAME_INDEX": 37}` | `{"EMPTY_CROP": 56, "NO_MATCHING_FRAME_INDEX": 37}` |
| `legacy_stride_keyframe_moto_coverage` | 49 / 100 *(with `bundle.persons`)* | **63 / 100** *(matches `100 - 37` Stage J predicate)* |
| `gapb_retention_keyframe_moto_coverage` | 98 / 100 | **100 / 100** |
| `gapb_sim_moto_descriptor_available` | 98 / 100 | **100 / 100** |
| `gapb_sim_unavailable_reasons` | `{"NO_MATCHING_FRAME_INDEX": 2}` | **`{}`** |
| `gapb_max_keyframes_per_camera_observed` | up to 37 *(TASK-07 audit)* | **24** (`<= 24` cap) |
| `gapb_streams_within_keyframe_cap` | 11 / 50 *(TASK-07 audit)* | **50 / 50** |
| `gapb_streams_synced_index_dict` | not enforced *(drifted)* | **50 / 50** |
| `gapb_stale_track_keyframe_refs` | unpruned | **0** |

All downstream metrics (`gap_b2_border_classification`, `gap_b3_evidence_and_gate`, `gap_b4_stagek_item_level_rider_association`, and `real_example_frames` `9/9`) matched TASK-06 identically.

### Production Metadata Replay Across All 50 Stage J Streams
Replaying frozen `results/phase12_stagej/*/*_detections.jsonl` and `*_track_frames.jsonl` directly through `apply_keyframe_retention_step()` (`test_task06a_stagej_50_streams_production_retention_invariants`):
- **`0`** impossible capacity conflicts across all 50 camera streams.
- **`<= 24`** retained keyframes on every frame of every stream (`pair_012/front` keeps `[152]` for `front_moto_0007` and `[159]` for `front_moto_0008`, evicting redundant earlier frames `30, 57, 27, 24, 21, 18, 15`).
- **`100 / 100`** final motorcycle tracklets, **`86 / 86`** rider subtracks, and **`130 / 130`** orphan tracks retain `>= 1` active keyframe.

---

## 6. Verification & Integrity Checks

1. **Targeted GAP-B + TASK-06A Suite:**  
   `/opt/homebrew/bin/python3.11 -m pytest tests/test_phase12_task06_gapb.py -v` -> **`21 passed in 6.90s`** (0 failed, 0 skipped).
2. **GAP-A & Phase 12E Suites:**  
   `/opt/homebrew/bin/python3.11 -m pytest tests/test_phase12_track_state_export.py tests/test_phase12e_association.py -v` -> **`55 passed in 3.07s`**.
3. **Full Test Suite:**  
   `/opt/homebrew/bin/python3.11 -m pytest tests/ -q` -> **`181 passed, 5 warnings in 95.97s`** (PyTorch deprecation warnings only, 0 failures, 0 skipped).
4. **Offline Determinism & Stage K Integrity:**  
   - `GAPB_EVAL_OUT_DIR=/tmp/task06a_gapb_verify1` vs `/tmp/task06a_gapb_verify2`: `diff -ru` exited `0` (`DETERMINISM_EXACT_MATCH`).
   - `scripts/reproducibility/evaluate.py --out /tmp/task06a_stagek` & `bootstrap.py --out /tmp/task06a_stagek`: exact baseline Stage K metrics reproduced (`head_det` pooled F1 `0.8161`, `association_rider_item_level` retained F1 `1.0` with `17/17` agreement, `events` F1 `0.8258`). Temporary directories cleaned after verification.
5. **Protected Artifact Checksums:**  
   - `data/dual_camera_test_v1/releases/dual_camera_gt_v1.0_ai_assisted_audited/checksums/release_checksums.sha256`: `ALL RELEASE CHECKSUMS OK`.
   - `results/phase12_integrated/checksums.sha256`: all 8 files `OK`.
   - `git status`: untouched across `data/`, `results/`, `configs/`, `src/tracking/`, `src/temporal/`, and pre-existing `docs/phase12/GEMINI_FLASH_TASK06_VALIDATION_REPORT.md`.

---

## 7. Known Operational Notes

- `evaluate_gapb_redesign.py` offline replay runs on CPU from frozen Stage J JSONL/CSV streams with deterministic synthetic `1280x720` frames (`real_example_frames` covers genuine `1280x720` crops from `docs/phase12/track_failure_examples/`).
- No full 25-pair video decode, YOLO inference, OCR run, or Kaggle GPU job was executed. Committed baseline files in `results/phase12_gapb_redesign/` remain at the TASK-06 snapshot for historical audit traceability alongside temporary-dir verification of TASK-06A.
