# GEMINI FLASH TASK-10 VALIDATION REPORT — Independent Audit of Bounded Real-Video Cue C (Gate B)

**Date:** 2026-11-12  
**Role:** Independent Engineering, Reproducibility & Data-Integrity Auditor (Gemini 3.8 Flash High)  
**Target Commit:** `963d238be70d6b53c1427bf63827c175f1b6028d`  
**Design Reference:** `docs/phase12/ARGON_TASK10_CUE_C_EXPERIMENT_DESIGN.md`  
**Implementation Report:** `docs/phase12/SONNET_TASK10_REAL_VIDEO_CUE_C_REPORT.md`  
**Evaluation Script:** `scripts/reproducibility/evaluate_task10_real_video_cue_c.py`  

---

## 1. Executive Verdict — **PASS WITH LIMITATIONS** (Gate B Cleared)

Independent verification confirms that **TASK-10 executes genuine PTS-indexed sequential CPU decoding on the bounded 4-pair real-video subset (`pair_005`, `pair_008`, `pair_009`, `pair_019`; `8/50` `.mp4` streams), preserves bit-exact deterministic CSV outputs and normalized JSON invariants, recovers `55/55` (`100.0%`) real-video `Cue C` descriptors from the Stage J `6/75` (`8.0%`) baseline, and proves `1280x720` vs `3840x2160` histogram invariance (`mean = 0.9971`)**.

The verdict is **`PASS WITH LIMITATIONS`** strictly because:
1. Coverage is bounded to **`4/25` video pairs** (`21/25` remaining pairs untested with real decoded frames per bounded compute constraints).
2. Real opposing-view imagery reveals genuine empirical limits of uncalibrated raw HSV+achromatic Bhattacharyya matching: a high unrelated-pair background overlap floor (`negative mean = 0.6641`, `median = 0.7306`), which nudges borderline unmatched `pair_005` (`0.6192` in `C0`) to `0.6197` in `C2 (720p, UNMATCHED)` and **`0.6203` in `C3 (4K, LINKED)`** (`+0.0003` over `min_association_score = 0.62`, false positive), while `C1 (A0 + 720p)` also false-linked `pair_005` (`0.6540`) due to `A0` starving the back rider tracklet.

---

## 2. Independent Verification & Checksum Integrity

1. **Git Provenance (`HEAD = 963d238be70d6b53c1427bf63827c175f1b6028d`):**
   - Strictly `8` added/updated TASK-10 files (`+5,335 / -0`), leaving `src/`, `configs/`, `data/`, `results/phase12_stagej/`, and `results/phase12_integrated/` completely untouched.
   - Untouched historical audit `docs/phase12/GEMINI_FLASH_TASK09_VALIDATION_REPORT.md` remains preserved.
2. **Immutable Checksums:**
   - GT v1.0: `annotation_checksums.sha256` **`39/39` OK**, `release_checksums.sha256` **`43/43` OK**.
   - Stage J (`results/phase12_stagej/`): **`381/381` OK** against `PACKAGE_MANIFEST.sha256`.
   - Stage K (`results/phase12_integrated/checksums.sha256`): **`8/8` OK**.
3. **Bounded Real-Video Decode Audit:**
   - Confirmed `BOUNDED_PAIRS = ["pair_005", "pair_008", "pair_009", "pair_019"]` decoded via `VideoReader(..., use_pts_index=True)` calling `get_frame_at_index()` sequentially forward with `cv2.VideoCapture.grab()` + `.retrieve()` (`0` `CAP_PROP_POS_FRAMES` seeks, `0` YOLO/ResNet runs).
   - Individual keyframe retention strictly `24/24` (`<= 24`) per camera on `8/8` streams for both `A0` and `A4`; shared decode union (`set(kf_a0) | set(kf_a4)`) decoded `201` total frames (`decode_call_count == unique_keyframes_decoded`, `cache_hit_count == 0` on `8/8` streams).

---

## 3. Independently Reproduced Quantitative Results (`/tmp/task10_verify_run1`, cleaned)

| Check / Metric | Committed (`results/phase12_task10/`) | Independent Replay (`/tmp/task10_verify_run1`) | Parity |
|---|---:|---:|---|
| `task10_pairwise_similarity.csv` (`27` lines) | SHA/Bytes matched | **Byte-identical (`cmp -s` = 0)** | **EXACT** |
| `task10_tracklet_descriptors.csv` (`242` lines) | SHA/Bytes matched | **Byte-identical (`cmp -s` = 0)** | **EXACT** |
| `task10_cue_c_results.json` (excl. runtime) | Deterministic payload | **Exact JSON equality** | **EXACT** |
| `wall_sec` / `peak_rss_mb` | `17.279s` / `1806.03 MB` | `16.917s` / `1818.17 MB` | Within tolerance |
| `checks` (`all_checks_passed`) | **`11/11` `True`** | **`11/11` `True`** | **PASS** |
| `tests/test_phase12_task10_cue_c.py` | — | **`3 passed` in `3.77s`** | **PASS** |
| Offline Suite (`--ignore=...real_video.py`) | — | **`187 passed` in `40.51s`** | **PASS** |
| Stage J Event Parity (`event_regression_task04.py`) | — | **`events compared=85 changed=0`** | **PASS** |

### Verified `C0..C4` Availability, Rejections & Resolution Fidelity
- **Stage J pre-GAP-B1 baseline (same `4` pairs):** `6/75` (`8.00%`: moto `2/21`, rider `3/16`, orphan `1/38`).
- **`C0_A4_no_cue_c`:** `0/55` (`NO_FRAME_IMAGES_PROVIDED: 55`).
- **`C1_A0_720p_kf`:** `75/75` (`100%`; moto `21/21` `0.9946`, rider `16/16` `0.9791`, orphan `38/38` `0.9624`); `4` frame rejections (`CROP_BELOW_MIN_RESOLUTION`: `1` rider, `3` orphan); misses `GAP-C` rider `pair_019/s2` and false-links `pair_005` (`0.6540`).
- **`C2_A4_720p_kf` (`1280x720`):** **`55/55` (`100%`)** (`21/21` moto `0.9932`, `17/17` rider `0.9804`, `17/17` orphan `0.9573`); `0` `OUT_OF_FRAME`, `0` `UNKNOWN_DIMS_MISMATCH`; `4` `CROP_BELOW_MIN_RESOLUTION` frame rejections (`1` rider, `3` orphan).
- **`C3_A4_4k_native` (`3840x2160`):** **`55/55` (`100%`)** (`21/21` moto `0.9889`, `17/17` rider `0.9812`, `17/17` orphan `0.9390`); `0` `OUT_OF_FRAME`, `0` `UNKNOWN_DIMS_MISMATCH`; `4` `CROP_MOTION_BLUR_BELOW_SHARPNESS` frame rejections (`1` moto, `3` rider).
- **`C4_A4_720p_none_dims_probe`:** **`52/55` (`94.5%`)** caught via `COORDINATE_SPACE_MISMATCH_UNKNOWN_NATIVE_DIMENSIONS` (`521` frame rejections); `3` tiny far-field tracks (`pair_009/back_moto_0006, 0007, 0008`, `x2 <= 1163.53, y2 <= 680.61`) stayed inside `[0..1280, 0..720]`, proving `frame_dimensions` forwarding is strictly mandatory.
- **`720p` vs `4K` Self-Similarity (`55` matched tracklets):** Composite `mean = 0.9971`, `min = 0.9857` (`moto_chassis n=21`: `0.9987` mean / `0.9920` min; `upper_torso n=40`: `0.9994` / `0.9967`; `cranial_cap n=43`: `0.9929` / `0.9651`); pairwise delta `mean = 0.0083`, `max = 0.0278`.
- **Opposing-View Discrimination (`5` GT positive vs `20` negative):** `720p` positive `mean = 0.7223` (`median = 0.7426`, `[0.5580, 0.7962]`) vs negative `mean = 0.6641` (`median = 0.7306`, `[0.2495, 0.8428]`); `4K` `0.7215` pos vs `0.6650` neg.

---

## 4. Documented Limitations & Gate B Authorization

1. **Knife-Edge Sensitivity & Background Overlap (`pair_005` `C3` = `0.6203`):** Uncalibrated raw HSV+achromatic overlap (`~0.55–0.85`) causes `C3 4K` (`Cue C = 0.6253`) to push `pair_005` across `min_association_score = 0.62` (`0.6192 -> 0.6203`), and leaves `C2 720p` at `0.6197`. Calibration / weight restraint is required before amplifying `weight_appearance_hsv`.
2. **Bounded Subset (`4/25` Pairs):** Remaining `21/25` pairs remain untested under live keyframe decoding.
3. **Minor Report Notes Verified:** Noted exact `A0 | A4` shared decode union (`201` total frames vs individual `<= 24/cam` `192` ceiling) and `pair_009` (`1x11` back motos, with `front_moto_0001 <-> back_moto_0001` already `LINKED` in Stage J and `C0` regardless of Cue C).

**Gate B Status:** **APPROVED (`PASS WITH LIMITATIONS`)**. Progression to **STAGE D (`TASK-11` — Dual-Camera Fusion Architecture & Fair Single-vs-Dual Ablation)** is authorized.
