# TASK-06 Implementation Report — Cross-Camera Tracklet Construction, Appearance Cue Recovery & Evaluation Integrity (GAP-B)

Baseline commit: `972dbb43b6f2c28b1c30c023fc7847c599b489af` (`main`).  
Architecture authority: `docs/phase12/ARGON_TASK05B_NEXT_RESEARCH_GAP_ASSESSMENT.md`.

---

## 1. Root Causes Addressed (GAP-B1 – GAP-B4)

1. **GAP-B1 (4K-to-720p Coordinate-Space Mismatch & Keyframe Starvation):**
   Stage J detections and tracks operate in native `3840x2160` coordinates while stored appearance keyframes are downscaled (`keyframe_max_width=1280` -> `1280x720`). Legacy unscaled cropping sliced 4K coordinates on 720p arrays (clamping to edge strips / `EMPTY_CROP`, `56/100` tracklets), while stride-only retention (`frame_index % stride == 0`, cap `24`) starved short or late-entering tracklets (`37/100` `NO_MATCHING_FRAME_INDEX`).
2. **GAP-B2 (Hardcoded `1920x1080` Border Assumption):**
   `TrackletBuilderConfig` defaulted to `frame_width_px=1920, frame_height_px=1080` and `DualCameraPipeline` never supplied camera frame dimensions, falsely flagging interior 4K detections (`x >= 1875` or `y >= 1035`) as `entry_near_border=True` / `exit_near_border=True`.
3. **GAP-B3 (Missing TASK-04 Evidence Propagation & Incomplete Cue-Count Gate):**
   `RiderTracklet`, `OrphanRiderTracklet`, and `RiderAssociationResult` exported only legacy decaying `phase10_snapshot_readonly` instead of TASK-04 `HelmetEvidenceAccumulator` outputs. Separately, `AssociationScorer.score_candidate_pair()` appended `INSUFFICIENT_ACTIVE_CUES` when active cues fell below `min_active_cues_required` without setting `passed_candidate_gate = False`.
4. **GAP-B4 (Stage K Rider Key Mismatch & Non-Portable Reproducibility Scripts):**
   Stage J association JSONs store per-item rider associations under `motorcycle_results[*].rider_associations` using `front_subtrack_id` / `back_subtrack_id` and `front_moto_track_id` / `back_moto_track_id`. Legacy `extract.py` did not extract `assoc_rider` and left item-level scoring `NOT COMPUTABLE`, while scripts under `scripts/reproducibility/` hardcoded VM paths (`~/workspace/research_helmet`, `/home/hatch/workspace/research_helmet`).

---

## 2. Implemented Architecture & Modified Files

### GAP-B1: Coordinate-Space Contract & Track-Aware Keyframe Retention
- **`src/association/coordinate_space.py` (new):** Implements `normalize_dimensions`, `scale_factors`, `CropResult`, `crop_native_box`, `is_box_near_border`, and `boxes_consistent_with_frame`. Maps native coordinates into stored-image coordinates via `(img_w / native_w, img_h / native_h)`, clamps sub-regions cleanly, rejects out-of-frame (`BOX_OUT_OF_FRAME`), zero-area (`ZERO_AREA_CROP`), invalid boxes (`INVALID_BOX`), and out-of-bounds boxes when native dimensions are omitted (`COORDINATE_SPACE_MISMATCH_UNKNOWN_NATIVE_DIMENSIONS`, tolerance `2.0 px`).
- **`src/association/appearance_features.py`:** Updated `_extract_subbox_crop`, `compute_crop_quality` (proportional `min_dim_scale = min(sx, sy)` with an `8 px` floor), and `extract_tracklet_appearance_descriptor(..., native_frame_dimensions=None)`. Enforces a tracklet-wide `all_tracklet_boxes` bounds check when `native_frame_dimensions is None` so early top-left frames of a 4K tracklet cannot be mis-cropped 1:1 on a 720p image.
- **`src/pipeline/dual_camera_pipeline.py`:** Added `CameraRunOutput.frame_dimensions`, `_observed_track_ids_at_step()`, and `should_retain_keyframe()` (enforcing strided sampling plus anchor retention for any newly observed track lacking a keyframe, evicting surplus keyframes only from multi-keyframe tracks within `keyframe_max_per_camera = 24`).

### GAP-B2: Dynamic Frame Dimensions & Border Status
- **`src/association/tracklet_builder.py`:** Changed `TrackletBuilderConfig.frame_width_px` and `frame_height_px` defaults to `None`; added `_resolve_frame_dimensions`, `_border_flags`, `BORDER_UNKNOWN_REASON = "BORDER_GEOMETRY_UNKNOWN_NO_FRAME_DIMENSIONS"`, and `BORDER_MISMATCH_REASON = "BORDER_COORDINATE_SPACE_MISMATCH"`; plumbed `frame_dimensions` through `extract_from_pipeline`, `build_motorcycle_tracklet`, and `build_orphan_tracklet`; updated `_combine_motorcycle_and_rider_appearance` to preserve `observed_frames_total`, `observed_frames_with_keyframe`, and `crop_rejection_counts` from `chassis_desc`.
- **`src/association/cross_camera_association.py`:** Added `front_frame_dimensions` and `back_frame_dimensions` to `associate_pipelines()` and optional `frame_width_px`/`frame_height_px` parsing in `from_yaml()`.
- **`src/pipeline/dual_camera_pipeline.py`:** Populates `frame_dimensions=(W, H)` from decoded frames in `run_camera()` and passes `front_frame_dimensions` / `back_frame_dimensions` in `run_association()`.

### GAP-B3: TASK-04 Evidence Propagation & Active-Cue Gate
- **`src/association/tracklet_models.py`:** Added `observed_frames_total`, `observed_frames_with_keyframe`, and `crop_rejection_counts` to `AppearanceDescriptorRecord`. Added `GENUINE_EVIDENCE_PROVENANCE`, `genuine_evidence_to_dict()`, `_GenuineHelmetEvidenceMixin` (`track_helmet_state`, `track_helmet_state_majority`, `track_helmet_state_conf_weighted`, `track_helmet_state_last_coupled`, `track_helmet_state_ema_final`, `coupled_observations_count`, etc.), and `genuine_helmet_evidence` / JSON key `"genuine_observation_helmet_evidence"` on `RiderTracklet` and `OrphanRiderTracklet`.
- **`src/association/tracklet_builder.py` & `src/association/rider_association.py`:** `_export_helmet_evidence()` copies the track's `HelmetEvidenceAccumulator.export(ema_ac)` (or an empty accumulator yielding `UNCERTAIN` with zero fabricated observations). `_readonly_phase10_dict()` includes `"genuine_observation_helmet_evidence"` on `RiderAssociationResult.front_phase10_readonly` / `back_phase10_readonly` while keeping `phase10_snapshot_readonly` untouched.
- **`src/association/association_scorer.py`:** Fixed `score_candidate_pair()` so `len(available_cues) < min_active_cues_required` (and any non-empty gate failure list) forces `passed_candidate_gate = False`.

### GAP-B4: Stage K Item-Level Rider Evaluation & Script Portability
- **`scripts/reproducibility/extract.py`:** Added portable repo-relative root resolution (`RESEARCH_HELMET_ROOT` / `Path(__file__).resolve().parents[2]`) with `os.PathLike` support (`load_gt(rel_dir=...)`, `load_pred(sj_dir=..., pairs=...)`), extracted `motorcycle_results[*].rider_associations` using `front_subtrack_id`/`back_subtrack_id`/`front_moto_track_id`/`back_moto_track_id` into `pred["assoc_rider"]`, and populated `pred["track_state_redesign"]`.
- **`scripts/reproducibility/evaluate.py` & `bootstrap.py`:** Wrapped evaluation into `run_evaluation(...)` defaulting `OUT` to `STAGEK_OUT_DIR` or `results/stagek_reproduced` (never mutating `results/phase12_integrated/`). Preserved `R["association_rider"]` verbatim and added `R["association_rider_item_level"]` (`retained_motorcycles` & `all_predicted_motorcycles`) plus `PP[pair]["assoc_rider"]` and optional `assoc_rider_f1` bootstrap CIs.
- **`scripts/reproducibility/*.py`:** Eliminated all hardcoded VM paths (`/home/hatch`, `~/workspace`, `/work/...`) across `analyze_tracks.py`, `annotate_pass1.py`, `build_reconciliation.py`, `evaluate.py`, `bootstrap.py`, `extract.py`, `extract_examples.py`, `measure_frame_counts.py`, `run_pass1_all.py`, and `validate_stagej.py`. Added `scripts/reproducibility/evaluate_gapb_redesign.py`.

---

## 3. Offline GAP-B Re-Evaluation Results (`results/phase12_gapb_redesign/`)

Replayed all **25 pairs** on CPU from frozen Stage J `*_detections.jsonl`, `*_coverage.csv`, and `*_sync_diagnostics.json` (`100` motorcycle tracklets, `86` rider subtracks, `130` orphan tracklets), plus **9 real `1280x720` PNG frames** from `docs/phase12/track_failure_examples/`. Zero video decoding and zero GPU inference were run; `1280x720` textured synthetic frames simulated end-to-end keyframe retention/cropping across the full 100-tracklet corpus.

| Dimension | Stage J Baseline | GAP-B Implementation | Artifact |
|---|---|---|---|
| **GAP-B1: Moto Appearance Available (corpus sim)** | `7 / 100` (`EMPTY_CROP: 56`, `NO_MATCHING_FRAME_INDEX: 37`) | **`98 / 100`** (`100%` of the 98 multi-frame `>=2`-frame moto tracklets; only 2 `observed_frames_total==1` tentative tracks `pair_012/front_moto_0007`, `front_moto_0008` missed keyframes) | `keyframe_and_appearance_audit.csv` |
| **GAP-B1: Keyframe Retention Coverage** | `49 / 100` (stride-only) | **`98 / 100`** (`should_retain_keyframe` within 24-frame cap) | `keyframe_and_appearance_audit.csv` |
| **GAP-B1: Real 720p PNGs (`docs/phase12/track_failure_examples/`)** | `0 / 9` usable (`EMPTY_CROP: 6`, `NO_MATCHING_FRAME_INDEX: 3`) | **`9 / 9` recovered** (`mean_crop_quality = 1.0` with `(3840, 2160)`); **`9 / 9`** explicitly rejected as `COORDINATE_SPACE_MISMATCH_UNKNOWN_NATIVE_DIMENSIONS` when dims omitted | `gapb_evaluation_results.json` |
| **GAP-B2: Border Classification (`entry` / `exit`)** | `entry=86`, `exit=75` (using hardcoded `1920x1080`) | **`entry=35`, `exit=33`** (**`51` entry + `42` exit false positives eliminated**, `0` false negatives; `100/100` flagged `BORDER_GEOMETRY_UNKNOWN_NO_FRAME_DIMENSIONS` when `frame_dimensions=None`) | `border_classification_comparison.csv` |
| **GAP-B3: Rider Subtracks (`N=86`)** | `HELMET=1, NO_HELMET=11, UNCERTAIN=74` | **`HELMET=20, NO_HELMET=66, UNCERTAIN=0`** (**`74 / 74` recovered**; `86/86` `phase10_snapshot_readonly` preserved) | `rider_tracklet_evidence_comparison.csv` |
| **GAP-B3: Orphan Tracklets (`N=130`)** | `HELMET=35, NO_HELMET=66, UNCERTAIN=29` | **`HELMET=38, NO_HELMET=91, UNCERTAIN=1`** (**`29 / 29` previously `UNCERTAIN` orphans recovered**; `pair_019/front_orphan_0001` is a 1:1 tie -> `UNCERTAIN`, `pair_004/front_orphan_0003` has `4 HELMET, 6 NO_HELMET` -> `NO_HELMET`) | `rider_tracklet_evidence_comparison.csv` |
| **GAP-B3: Cue-Count Gate Audit (`N=94` candidates)** | Unenforced gate possibility | **`0` violations** (`insufficient_active_cues_gate_violations = 0`) | `gapb_evaluation_results.json` |
| **GAP-B4: Stage K Item-Level Rider Assoc (retained motos)** | Not computable in Stage K | **`P=1.0, R=1.0, F1=1.0`** (`TP=4, FP=0, FN=0`; `17/17` exact outcome agreement; `coverage=1.0`) | `stagek_amended_results.json`, `rider_association_item_level.csv` |
| **GAP-B4: Stage K Item-Level Rider Assoc (all predicted motos)** | Not computable in Stage K | **`P=1.0, R=1.0, F1=1.0`** (`20` predicted items, `3` excluded parent moto records) | `stagek_amended_results.json` |

Generated artifacts under `results/phase12_gapb_redesign/`:
- `gapb_evaluation_results.json`
- `stagek_amended_results.json`
- `border_classification_comparison.csv`
- `rider_tracklet_evidence_comparison.csv`
- `keyframe_and_appearance_audit.csv`
- `rider_association_item_level.csv`

---

## 4. Verification & Test Suite

1. **Targeted Suites:**
   `/opt/homebrew/bin/python3.11 -m pytest tests/test_phase12_task06_gapb.py tests/test_phase12_track_state_export.py tests/test_phase12e_association.py -v`
   -> **`66 passed in 6.55s`** (`11` new GAP-B tests + `16` TASK-04 tests + `39` Phase 12E association tests).
2. **Full Repository Suite:**
   `/opt/homebrew/bin/python3.11 -m pytest -q`
   -> **`171 passed, 5 warnings in 108.60s`** (warnings are pre-existing upstream PyTorch/EasyOCR quantization deprecations).
3. **Standalone Stage K Reproducibility Check:**
   `python3.11 scripts/reproducibility/evaluate.py --out /tmp/stagek_gapb_verify && python3.11 scripts/reproducibility/bootstrap.py --out /tmp/stagek_gapb_verify`
   -> Clean execution; `association_rider_item_level` exact metrics above and `assoc_rider_f1` bootstrap 95% CI `[1.0, 1.0]` (`mean=0.988`); all legacy metrics (`association_motorcycle`, `association_orphan`, `events`, `detection_head_operating_point`, `IDF1`) identically reproduced.

---

## 5. Protected-Artifact Integrity & Limitations

- **Protected artifacts unchanged:** Frozen GT release (`data/dual_camera_test_v1/releases/dual_camera_gt_v1.0_ai_assisted_audited/`), `synchronization_metadata.json`, `results/phase12_stagej/`, `results/phase12_integrated/`, and source videos remain untouched.
- **Limitations:** Full 25-pair offline Cue C descriptor validation used CPU replay of Stage J detections paired with synthetic `1280x720` textured frames (alongside 9 real `1280x720` PNG frames from `docs/phase12/track_failure_examples/`) because full 50-video GPU re-inference was explicitly out of scope for TASK-06. Real video re-inference with GAP-B enabled remains an optional downstream validation run.
