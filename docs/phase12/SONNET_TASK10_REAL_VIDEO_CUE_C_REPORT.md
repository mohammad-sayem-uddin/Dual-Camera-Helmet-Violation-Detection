# SONNET TASK-10 REAL-VIDEO CUE C REPORT
## Bounded Empirical Validation on Real Compressed 4K/720p Dual-Camera Streams

**Date:** 2026-11-12  
**Author:** Claude Sonnet 5.5 (Implementation & Empirical Evaluation Engineer)  
**Architecture Design:** `docs/phase12/ARGON_TASK10_CUE_C_EXPERIMENT_DESIGN.md`  
**Evaluation Script:** `scripts/reproducibility/evaluate_task10_real_video_cue_c.py`  
**Artifacts Directory:** `results/phase12_task10/` (`task10_cue_c_results.json`, `task10_tracklet_descriptors.csv`, `task10_pairwise_similarity.csv`)  

---

## 1. Executive Summary

Using PTS-indexed sequential decoding (`VideoReader(..., use_pts_index=True)`) on the bounded **4-pair (`8 / 50` videos)** subset (`pair_005`, `pair_008`, `pair_009`, `pair_019`) with cached Stage J JSONL detections (zero full-corpus decoding, zero YOLO/ResNet model runs), all **`11/11` technical and coordinate-contract integrity checks pass (`all_checks_passed = true`)** (`Wall: 17.279s`, `Peak RSS: 1806.03 MB`):

1. **Stage J -> TASK-10 Recovery:** Across these `4` pairs, real-video Cue C descriptor availability rose from **`6 / 75` (`8.00%`)** in frozen Stage J (pre-GAP-B1/GAP-C; corrupted top-left 720p crops) to **`55 / 55` (`100.0%`)** under `C2 (GAP-C A4 + 1280x720 Keyframes)` and `C3 (GAP-C A4 + 3840x2160 Native)` (`21/21` motorcycles, `17/17` riders, `17/17` orphans).
2. **`1280x720` vs `3840x2160` Coordinate-Space Invariance (`H1` Verified):** Across all `55` matched tracklets, production `720p` resized keyframes achieved **`0.9971` mean (`0.9857` min)** composite Bhattacharyya self-similarity against full `4K` native frames (`0.9987` chassis, `0.9994` torso, `0.9929` cranial), with **`0.0083` mean (`0.0278` max)** cross-camera score delta, **`0` `OUT_OF_FRAME`**, and **`0` `UNKNOWN_DIMS_MISMATCH`** errors.
3. **Unknown-Dimensions Safety Guard (`C4` Negative Control Verified):** Omitting `native_frame_dimensions` on `720p` keyframes triggered `COORDINATE_SPACE_MISMATCH_UNKNOWN_NATIVE_DIMENSIONS` on **`52 / 55` (`94.5%`)** real tracklets (`521` frame rejections); the remaining `3` tiny far-field motorcycles (`pair_009/back_moto_0006, 0007, 0008`) fell within `[0..1280, 0..720]`, underscoring why explicit `frame_dimensions` wiring is always mandatory.
4. **Honest Opposing-View Limitations & Boundary Sensitivity Discovered (`H3`):**
   - Across `5` GT positive pairs vs `20` same-scene negative/distractor pairs, Cue C averages **`0.7223` vs `0.6641`** (`720p`, median `0.7426` vs `0.7306`) — providing helpful disambiguation on `pair_019` (`+0.36` vs wrong convoy distractor `back_moto_0001`, reducing wrong composite `0.8608 -> 0.7748`, and `+0.1869` between recovered driver `s2 <-> s0` vs wrong pillion `s0 <-> s0`), while showing modest separation when clothing/background are neutral grey/black or when face vs back-of-head flips lower `cranial_cap` (`pair_009` orphan `cranial_cap = 0.2685`, `upper_torso = 0.7509`).
   - **Critical Finding (`pair_005` `C3 4K` False Positive):** On `pair_005` (`UNMATCHED` non-match), adding uncalibrated raw Bhattacharyya Cue C (`0.6253` at `4K`, actually *below* the `0.6641` unrelated negative mean!) pushed the borderline `0.6192` (`C0`) score to `0.6197` in `C2 (720p, UNMATCHED)` and **`0.6203` in `C3 (4K, LINKED)`** — crossing `min_association_score = 0.62` by **`+0.0003`** into a false positive! This proves raw Bhattacharyya overlap (`~0.55–0.73` background floor) requires empirical calibration before increasing `weight_appearance_hsv` above `0.15`.

---

## 2. Decode Bounds & Stream Integrity

Every arm strictly obeyed `keyframe_max_per_camera = 24` (`keyframe_stride = 3`) and exact sequential single-pass PTS decoding (`decode_call_count == unique_keyframes_decoded` on `8 / 8` streams):

| Pair | Front Frames (Total) | Back Frames (Total) | `A4` Retained (`F / B`) | `A0` Retained (`F / B`) | Union Decoded (`F / B`) | Decode Calls (`F / B`) |
|---|---:|---:|---:|---:|---:|---:|
| `pair_005` | `206` | `352` | `24 / 24` | `24 / 24` | `24 / 24` | `24 / 24` |
| `pair_008` | `156` | `270` | `24 / 24` | `24 / 24` | `25 / 24` | `25 / 24` |
| `pair_009` | `232` | `415` | `24 / 24` | `24 / 24` | `24 / 25` | `24 / 25` |
| `pair_019` | `372` | `231` | `24 / 24` | `24 / 24` | `28 / 27` | `28 / 27` |

*(Note: Individual `A0` and `A4` runs never exceeded `24` retained frames per camera on `8/8` streams; only the shared decode cache took the set union `A0 | A4` to avoid opening any video twice.)*

---

## 3. Descriptor Availability & Quality Across Experimental Arms

| Arm | Motorcycles | Riders | Orphans | Mean Crop Quality (M / R / O) | Unavailable Reasons | Frame Crop Rejections |
|---|---|---|---|---|---|---|
| **Stage J (`pre-GAP-B1/C`)** | `2/21` (`9.5%`) | `3/16` (`18.8%`) | `1/38` (`2.6%`) | — *(corrupted 720p coords)* | `EMPTY_CROP`, `NO_MATCHING_FRAME_INDEX` | Unmapped 4K coords on 720p |
| **`C0` (`A4`, no keyframes)** | `0/21` (`0%`) | `0/17` (`0%`) | `0/17` (`0%`) | — | `NO_FRAME_IMAGES_PROVIDED: 55` | `{}` |
| **`C1` (`A0`, `720p`)** | `21/21` (`100%`) | `16/16` (`100%`) | `38/38` (`100%`) | `0.9946 / 0.9791 / 0.9624` | `{}` | `CROP_BELOW_MIN_RESOLUTION: 4` |
| **`C2` (`A4`, `720p`)** | **`21/21` (`100%`)** | **`17/17` (`100%`)** | **`17/17` (`100%`)** | **`0.9932 / 0.9804 / 0.9573`** | **`{}`** | `CROP_BELOW_MIN_RESOLUTION: 4` |
| **`C3` (`A4`, `4K`)** | **`21/21` (`100%`)** | **`17/17` (`100%`)** | **`17/17` (`100%`)** | **`0.9889 / 0.9812 / 0.9390`** | **`{}`** | `CROP_MOTION_BLUR_BELOW_SHARPNESS: 4` |
| **`C4` (`A4`, `720p`, `dims=None`)** | `3/21` (`14.3%`) | `0/17` (`0%`) | `0/17` (`0%`) | `1.000 / 0.0 / 0.0` | `COORDINATE_SPACE_MISMATCH_UNKNOWN_NATIVE_DIMENSIONS: 52` | `UNKNOWN_DIMS_MISMATCH: 521` |

**`GAP-C` Impact (`C1` vs `C2`):** `A0` missed `pair_019/front_moto_0002_rider_s2` (stalled behind abstentions) and exported `38` orphan tracks (mostly 1-hit low-conf noise) across these 4 pairs, whereas `A4` (`C2`) recovered the `17` genuine coupled rider subtracks (`100%` valid Cue C torso+cranial descriptors) and trimmed orphans to `17` (`100%` Cue C available).

---

## 4. `720p` vs `4K` Resolution Invariance & Pairwise Discrimination

### 4.1 Per-Zone Resolution Self-Similarity (`C2 720p` vs `C3 4K`)
- `moto_chassis` (`n = 21`): `mean = 0.9987`, `min = 0.9920`
- `upper_torso` (`n = 40`): `mean = 0.9994`, `min = 0.9967`
- `cranial_cap` (`n = 43`): `mean = 0.9929`, `min = 0.9651`
- `composite` (`n = 55`): **`mean = 0.9971`**, **`min = 0.9857`**
- Cross-camera score stability (`|CueC_720p - CueC_4K|`): **`mean = 0.0083`**, **`max = 0.0278`**.

### 4.2 Key Pairwise Real-Video Results (`results/phase12_task10/task10_pairwise_similarity.csv`)

| Pair | Entities (`Front <-> Back`) | Ground-Truth Role | Cue C (`720p` / `4K`) | Zone Breakdown (`720p`: Cranial / Torso / Chassis) | Composite (`C0 -> C2 -> C3`) | Outcome (`C0 / C2 / C3`) |
|---|---|---|---|---|---|---|
| `pair_005` | `front_moto_0001 <-> back_moto_0001` | `NON_MATCH` (`UNMATCHED`) | `0.6216 / 0.6253` | `0.5190 / 0.5633 / 0.8497` | `0.6192 -> 0.6197 -> 0.6203` | `UNM / UNM / LINKED†` |
| `pair_005` | `front_orphan_0002 <-> back_moto_0001_s0` | Asymmetric negative | `0.6565 / 0.6596` | `0.6053 / 0.6907 / —` | — | `NOT_L / NOT_L / NOT_L` |
| `pair_008` | `front_moto_0001 <-> back_moto_0001` | **`GT_LINKED`** | `0.7962 / 0.7988` | `0.6203 / 0.8868 / 0.8441` | `0.9763 -> 0.9445 -> 0.9450` | **`LNK / LNK / LNK`** |
| `pair_008` | `rider_s0 <-> back_s0` | **`GT_LINKED_RIDER`** | `0.7802 / 0.7850` | `0.6203 / 0.8868 / —` | — | **`LNK / LNK / LNK`** |
| `pair_008` | `rider_s1 <-> back_s0` | Same-moto wrong slot | `0.8428 / 0.8549` | `0.8428 / — / —` | — | `NOT_L / NOT_L / NOT_L` |
| `pair_009` | `front_orphan_0002 <-> back_orphan_0001` | **`GT_LINKED_ORPHAN`** | `0.5580 / 0.5558` | `0.2685 / 0.7509 / —` | — | **`LNK / LNK / LNK`** |
| `pair_019` | `front_moto_0002 <-> back_moto_0002` | **`GT_LINKED`** | `0.7346 / 0.7304` | `0.7721 / 0.7230 / 0.7104` | `0.8592 -> 0.8372 -> 0.8364` | **`LNK / LNK / LNK`** |
| `pair_019` | `front_moto_0002 <-> back_moto_0001` | Same-window distractor | `0.3731 / 0.3762` | `— / — / 0.3731` | `0.8608 -> 0.7748 -> 0.7753` | `UNM / UNM / UNM` |
| `pair_019` | `front_moto_0001 <-> back_moto_0001` | Negative distractor | `0.2495 / 0.2558` | `— / — / 0.2495` | `0.2072 -> 0.2146 -> 0.2157` | `UNM / UNM / UNM` |
| `pair_019` | `front_moto_0003 <-> back_moto_0002` | Negative distractor | `0.7323 / 0.7450` | `0.6081 / 0.8396 / 0.6880` | `0.7275 -> 0.7284 -> 0.7306` | `UNM / UNM / UNM` |
| `pair_019` | `moto2_s2 <-> back_moto2_s0` | **`GAPC_RECOVERED_DRV`** | `0.7426 / 0.7374` | `0.7721 / 0.7230 / —` | — | **`LNK / LNK / LNK`** |
| `pair_019` | `moto2_s0 <-> back_moto2_s0` | Wrong co-rider | `0.5557 / 0.5835` | `0.4880 / 0.6008 / —` | — | `NOT_L / NOT_L / NOT_L` |

*†Note on `pair_005/front_moto_0001 <-> back_moto_0001`: borderline unmatched pair (`0.6192` in `C0`) stays `UNMATCHED` in `C2 (720p)` (`0.6197`), crosses `min_association_score = 0.62` by `+0.0003` into a false `LINKED` in `C3 (4K)` (`0.6203`, `Cue C = 0.6253`), and also false-linked in `C1 (A0+720p)` at `0.6540` (`Cue C = 0.8161`) because `A0` starved the back rider tracklet.*  
*Background note on `pair_009/front_moto_0001 <-> back_moto_0001`: both Stage J (`0.6707`) and `C0` (`0.8743`) already linked this pair without Cue C (`C2 = 0.8681`, `C3 = 0.8675`), whereas GT excluded `back_moto_0001` in Stage E recalculation (`cross_camera_identity_links.csv` L83).*

---

## 5. Verification & Architectural Recommendations for TASK-11

- **Tests:** Executed `tests/test_phase12_task10_cue_c.py` alongside the full offline suite (`pytest tests/ -q --ignore=tests/test_phase12_pipeline_real_video.py`).
- **Two Concrete Engineering Takeaways:**
  1. **`1280x720` Keyframes Are Validated for Production Use:** Downscaling retained `<= 24` keyframes to `1280` px width preserves Cue C within `< 0.01` average cross-camera difference vs `4K` while cutting memory substantially and eliminating `UNKNOWN_DIMS_MISMATCH` and `OUT_OF_FRAME` errors once `frame_dimensions` is passed.
  2. **Appearance Calibration Caution:** Raw HSV+achromatic Bhattacharyya overlap carries a `~0.55–0.73` background floor across asphalt/dark gear in opposing views (`moto_chassis` `~0.85` on `pair_005`). Keep `weight_appearance_hsv = 0.15` conservative (or subtract/calibrate background expectation) and do **not** treat raw Cue C alone as strong identity confirmation without temporal+role corroboration.
