# ARGON TASK-10 — BOUNDED REAL-VIDEO CUE C EXPERIMENT DESIGN
## Coordinate-Space Fidelity, Keyframe Retention & Opposing-View HSV+Achromatic Discrimination Under Real H.264/HEVC Video

**Date:** 2026-11-12  
**Author:** Gemini Argon (Senior Computer Vision Research Architect)  
**Verified Baseline:** `HEAD = 8cc3c96` (`TASK-09A` + `TASK-09B` Gate A `PASS WITH LIMITATIONS`)  
**Target Modules:** `src/video/video_reader.py`, `src/pipeline/dual_camera_pipeline.py`, `src/association/coordinate_space.py`, `src/association/appearance_features.py`, `src/association/tracklet_builder.py`, `src/association/cross_camera_association.py`  

---

## 1. Scientific Problem & Experimental Purpose

Across `TASK-06`, `TASK-06A`, `TASK-09A`, and `TASK-09B`, all Cue C (`appearance_hsv`) checks executed either synthetic `1280x720` checkerboard textures (`scripts/reproducibility/evaluate_gapb_redesign.py`) or disabled keyframes (`frame_images_by_index=None` in `evaluate_gapc_tracking.py`). Consequently, **real-video Cue C extraction remains unverified** (`CUE_C_REAL_VIDEO_NOT_EXECUTED`):

1. **Resolution & Coordinate Scaling (`3840x2160` -> `1280x720`):** Production `DualCameraPipeline` downscales retained keyframes via `_resize_keyframe(frame, max_width=1280)` (`(1280, 720)` for `3840x2160` source footage, `scale = 1/3`) and relies on `crop_native_box()` + proportional `min_dim_scale` (floored at `8 px`) and `min_laplacian_sharpness = 12.0` (`configs/cross_camera_association_config.yaml`). Real compressed footage tests whether downscaling to `720p` degrades Laplacian sharpness, rejects valid small head/orphan crops, or shifts HSV+achromatic histograms vs native `4K`.
2. **Upstream Tracking Coupling (`A0` vs `A4` GAP-C):** Cue C assigns weights `0.30` (`cranial_cap`), `0.45` (`upper_torso`), and `0.25` (`moto_chassis`), deriving torso boxes from `RiderSubTrack` head boxes (`_combine_motorcycle_and_rider_appearance`). We test whether GAP-C (`A4`) materially increases real-video `RiderTracklet` and motorcycle Cue C descriptor quality/availability compared to legacy `A0`.
3. **Opposing-View Discrimination (Front vs. Back Camera):** Front and Back cameras view traffic from opposite longitudinal directions under different exposure/backlighting and H.264 vs HEVC compression. Real pairs measure true empirical separation between GT `LINKED` pairs and non-matching scene distractors.

---

## 2. Strictly Bounded Sample & Resource Budget

Per project constraints (**no decoding all 25 pairs, no full-video detector inference, CPU-only Apple Silicon execution**), we select a **4-pair bounded diagnostic subset (`8 / 50` videos)**:

| Pair | Container Info (`data/dual_camera_test_v1`) | Sync Quality / Timing | Selected Research Role in TASK-10 |
|---|---|---|---|
| **`pair_005`** | `front`: 206f H.264, `back`: 352f HEVC (`3840x2160`) | `HIGH` (`+70 ms` drift) | **Unmatched Negative & Asymmetric Orphan Case:** `UNMATCHED` motorcycles + short `9`-hit `front_orphan_0002` vs `118`-hit `back_moto_0001_rider_s0` (tests negative appearance separation & false-link immunity). |
| **`pair_008`** | `front`: 156f H.264, `back`: 270f HEVC (`3840x2160`) | `MEDIUM` (`-76 ms` drift) | **Single-Vehicle Positive & Linked Rider Case:** GT `LINKED` motorcycle (`front_moto_0001 <-> back_moto_0001`) + `LINKED` driver (`s0 <-> s0`) + `s1` pillion. |
| **`pair_009`** | `front`: 232f H.264, `back`: 415f HEVC (`3840x2160`) | `HIGH` | **Genuine Orphan Positive & Multi-Distractor Case:** GT `LINKED` **non-motorcycle orphan** (`front_orphan_0002 <-> back_orphan_0001`) + `1` Front moto vs `4` Back unmatched motos (`1x4` negative pairs). |
| **`pair_019`** | `front`: 372f H.264, `back`: 231f HEVC (`3840x2160`) | `MEDIUM` (`-16 ms` drift, `eps=8 ms`) | **Multi-Motorcycle Convoy (`1 TP vs 5 TN`) & GAP-C Recovery Case:** `3` Front motos x `2` Back motos (`front_moto_0002 <-> back_moto_0002` GT `LINKED`, `5` same-scene distractors), `GAP-C` recovered rider `s2 <-> s0`, and `>= 10` orphans. Validated PTS index in `test_phase12_pipeline_real_video.py`. |

### Decoding Discipline
- **Zero full-corpus decode & zero YOLO/ResNet model execution:** Detections come verbatim from cached Stage J JSONL (`results/phase12_stagej/`).
- **PTS-indexed selective frame decode only:** `VideoReader(..., use_pts_index=True)` advances forward with lightweight `cv2.VideoCapture.grab()` and invokes `.retrieve()` **exclusively** on the `<= 24` frame indices per camera selected by production `apply_keyframe_retention_step(..., keyframe_stride=3, keyframe_max_per_camera=24)` (maximum `192` retrieved frames across all `8` `.mp4` streams; actual retained count measured and logged). Readers close cleanly after each camera stream.

---

## 3. Experimental Arms & Evaluation Protocol

All `5` arms operate on the exact same `4` pairs (`8` streams) and cached Stage J detections + container PTS maps:

1. **`C0` (`A4`, Keyframes Disabled):** `GAP-C A4` tracker with `frame_images_by_index=None` (`Cue C = N/A`, pure non-appearance baseline).
2. **`C1` (`A0`, `720p_keyframe` `1280x720`):** Legacy `A0` tracker + production downscaled keyframes (`_resize_keyframe(frame, 1280)` -> `(1280, 720)`) with `native_frame_dimensions=(3840, 2160)`. Tests whether `A0` Back-camera subtrack abstentions starve real-video `RiderTracklet` Cue C.
3. **`C2` (`A4`, `720p_keyframe` `1280x720`):** Production target regime: `GAP-C A4` + `_resize_keyframe(frame, 1280)` -> `(1280, 720)` + `native_frame_dimensions=(3840, 2160)` (`scale = 1/3`).
4. **`C3` (`A4`, `4K_native` `3840x2160`):** Unscaled reference regime: `GAP-C A4` + native `(3840, 2160)` frames on the exact same keyframe indices (`scale = 1.0`). Direct test of `720p` vs `4K` invariance.
5. **`C4` (`A4`, `720p_keyframe`, `native_frame_dimensions=None` Negative Control):** Verifies on real compressed `3840x2160`->`1280x720` inputs that omitting `native_frame_dimensions` safely triggers `REJ_UNKNOWN_DIMS_MISMATCH` rather than silently mis-cropping `4K` boxes on `720p` images.

---

## 4. Verification Metrics & Quantitative Checks (`results/phase12_task10/`)

1. **Keyframe & Coordinate Contract Integrity:**
   - Strictly `<= 24` retained keyframes/camera on `8/8` streams; `decode_call_count <= retained_keyframes` per video; zero `OOB` boxes in `A4`.
   - Zero `UNKNOWN_DIMS_MISMATCH` and zero `OUT_OF_FRAME` rejections across all valid crops in `C2` and `C3` (`C4` explicitly catches dimension mismatch).
   - Full `crop_rejection_counts` breakdown (`CROP_BELOW_MIN_RESOLUTION`, `LOW_SHARPNESS_LAPLACIAN`, `NO_MATCHING_FRAME_INDEX`, etc.) at `720p` vs `4K`.
2. **Real-Video Descriptor Availability (`C1` vs `C2` vs `C3`):**
   - Separate availability counts for `MotorcycleTracklet`, `RiderTracklet`, and `OrphanRiderTracklet`.
3. **Cross-Resolution Descriptor Fidelity (`C2 720p` vs `C3 4K`):**
   - Self-similarity (`bhattacharyya_histogram_similarity`) of each tracklet's `720p` descriptor vs its `4K` descriptor (`moto_chassis`, `cranial_cap`, `upper_torso`, composite Cue C), plus max/mean pairwise score delta `|S_720p - S_4k|`.
4. **Empirical Cross-Camera Similarity Discrimination:**
   - Exact Cue C scores (`score_appearance_similarity`) on every GT `LINKED` pair (motorcycles, riders, orphan `pair_009`) vs every same-scene `UNMATCHED` distractor pair (`pair_005`, `pair_009`, `pair_019`), reported honestly with no selective filtering (explicitly reporting if opposing-view illumination/clothing variation yields modest separation).
5. **End-to-End Association Parity & Stability (`C0` vs `C2` vs `C3`):**
   - Zero `NaN`/`Inf` scores; zero loss of GT `LINKED` pairs (`pair_008` moto+driver, `pair_009` orphan, `pair_019` moto+driver); zero false links introduced on `pair_005` / `pair_009` / `pair_019`.
