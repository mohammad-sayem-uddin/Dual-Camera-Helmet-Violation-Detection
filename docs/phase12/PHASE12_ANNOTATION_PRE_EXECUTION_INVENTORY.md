# PHASE 12 — ANNOTATION PRE-EXECUTION INVENTORY

**Project:** Dual-Camera Helmet Violation Detection (`research_helmet`)  
**Mission:** Complete AI-assisted annotation of all 25 synchronized dual-camera pairs  
**Dataset label:** `AI_ASSISTED_AUDITED` (AI-assisted; NOT human-verified)  
**Inventory date:** 2026-11-08  
**Repository state:** Mac `/Users/sayemuddin/Desktop/research_helmet` == GitHub `main` @ `6aba04434af80cf0263de031a3c06488bebec29f` (Phase 12E Review Correction 3). VM working mirror extracted from the GitHub codeload tarball of that commit.

## 1. Verification summary

- Expected: 25 synchronized pairs / 50 source videos. **Found: exactly 25 front + 25 back = 50 videos. No discrepancy.**
- Front folder: `data/dual_camera_test_v1/front/pair_001_front.mp4` … `pair_025_front.mp4` (25 files, verified by direct listing on the Mac).
- Back folder: `data/dual_camera_test_v1/back/pair_001_back.mp4` … `pair_025_back.mp4` (25 files, verified by direct listing on the Mac).
- All 25 pairs are `CONFIRMED` in the Phase 11 `pair_manifest.csv`. Sync quality: HIGH ×22, MEDIUM ×3 (pair_008, pair_019, pair_022). Near-frame pairs (residual 18–38 ms): pair_004, pair_005, pair_008, pair_015. Sub-frame pairs: the remaining 21.
- Total decoded frames (Phase 12C forensic audit; OpenCV sequential decode counts matched container counts exactly for all 50 streams): **15,241 frames = 6,557 front + 8,684 back**. Total duration 274.33 s; common synchronized overlap 208.11 s.
- All 50 videos are 3840×2160. Front: H.264, ~60 fps nominal, AAC 48 kHz stereo. Back: HEVC, AAC 48 kHz mono; 59.94 fps for most pairs, VFR for pair_013/pair_014 back, 29.97 fps for pair_015–018 back. Forensically VFR streams (irregular PTS intervals): `pair_001_front`, `pair_013_back`, `pair_014_back`.
- Source-file note: source front files `11.mp4` and `16.mp4` exist in the original capture folder but belong to NO pair (pair_011 uses source `12.mp4`; pair_016 uses `18.mp4`). They are outside this mission's 50-video scope.

## 2. Transfer and integrity (VM working mirror)

- 43/50 videos transferred Mac → VM and **SHA-256 verified byte-identical** against `pair_manifest.csv` (43/43 match).
- 7 videos exceed the 50 MiB per-file transfer limit and are pending Mac-side splitting, after which they will be reassembled and hash-verified before any processing: front pair_013 (60,641,444 B), pair_016 (82,626,269 B), pair_019 (56,715,060 B), pair_021 (52,647,179 B), pair_024 (74,072,602 B); back pair_013 (63,695,588 B), pair_024 (76,986,346 B). **No annotation pass will run on a pair until both of its videos are hash-verified on the VM.**
- Originals on the Mac are never modified, renamed, recompressed, or deleted; the VM works on verified copies.

## 3. Per-video inventory

| Pair | Camera | File | Bytes | Codec | Resolution | Eff. FPS | Frames (container = decoded) | Duration (s) | VFR | Sync quality | Residual (ms) | Offset (s) | SHA-256 (manifest) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pair_001 | front | pair_001_front.mp4 | 31534841 | h264 | 3840x2160 | 59.7232 | 206 | 3.4325 | True | HIGH | 3.0 | -0.205 | 995d1fa69ea4bb7a… |
| pair_001 | back | pair_001_back.mp4 | 41859002 | hevc | 3840x2160 | 59.9243 | 397 | 6.608333 | False | HIGH | 3.0 | -0.205 | 71684f29a864ea9b… |
| pair_002 | front | pair_002_front.mp4 | 47130244 | h264 | 3840x2160 | 60.0156 | 313 | 5.198644 | False | HIGH | 3.0 | -1.055 | b902c441e1099d5a… |
| pair_002 | back | pair_002_back.mp4 | 44551021 | hevc | 3840x2160 | 59.929 | 423 | 7.041667 | False | HIGH | 3.0 | -1.055 | 54db57f4a518a849… |
| pair_003 | front | pair_003_front.mp4 | 35342431 | h264 | 3840x2160 | 60.0149 | 234 | 3.882367 | False | HIGH | 3.0 | 0.665 | 17f2ace3dccb83d2… |
| pair_003 | back | pair_003_back.mp4 | 46048981 | hevc | 3840x2160 | 59.9311 | 436 | 7.258333 | False | HIGH | 3.0 | 0.665 | 8157d17eaa0da0b9… |
| pair_004 | front | pair_004_front.mp4 | 26941095 | h264 | 3840x2160 | 60.0151 | 179 | 2.965922 | False | HIGH | 18.0 | -1.285 | dff63a5e00eb931d… |
| pair_004 | back | pair_004_back.mp4 | 21378866 | hevc | 3840x2160 | 59.9401 | 201 | 3.336667 | False | HIGH | 18.0 | -1.285 | e36871461fc04117… |
| pair_005 | front | pair_005_front.mp4 | 31123415 | h264 | 3840x2160 | 60.0156 | 206 | 3.415778 | False | HIGH | 35.0 | 0.54 | 0c041d229095c73f… |
| pair_005 | back | pair_005_back.mp4 | 36967910 | hevc | 3840x2160 | 59.9317 | 352 | 5.856667 | False | HIGH | 35.0 | 0.54 | 85ef3af22f7e8d31… |
| pair_006 | front | pair_006_front.mp4 | 20134145 | h264 | 3840x2160 | 60.0149 | 135 | 2.232778 | False | HIGH | 6.0 | -0.045 | 758a97a08e5912e0… |
| pair_006 | back | pair_006_back.mp4 | 27097721 | hevc | 3840x2160 | 59.9292 | 255 | 4.238333 | False | HIGH | 6.0 | -0.045 | 883a57e7375d1f9a… |
| pair_007 | front | pair_007_front.mp4 | 23689481 | h264 | 3840x2160 | 60.0144 | 157 | 2.599378 | False | HIGH | 4.0 | 0.415 | 20177584b3b6ab76… |
| pair_007 | back | pair_007_back.mp4 | 25503026 | hevc | 3840x2160 | 59.9245 | 239 | 3.971667 | False | HIGH | 4.0 | 0.415 | 30e2bd2d5fdda83d… |
| pair_008 | front | pair_008_front.mp4 | 23785072 | h264 | 3840x2160 | 60.0152 | 156 | 2.582678 | False | MEDIUM | 38.0 | 0.755 | a09374101a00268b… |
| pair_008 | back | pair_008_back.mp4 | 28983747 | hevc | 3840x2160 | 59.9332 | 270 | 4.488333 | False | MEDIUM | 38.0 | 0.755 | 5349f3c472347f7d… |
| pair_009 | front | pair_009_front.mp4 | 35173385 | h264 | 3840x2160 | 60.0154 | 232 | 3.849011 | False | HIGH | 3.0 | 0.225 | b49f5204a5ef7644… |
| pair_009 | back | pair_009_back.mp4 | 43992563 | hevc | 3840x2160 | 59.9276 | 415 | 6.908333 | False | HIGH | 3.0 | 0.225 | 69dc9fa55d697590… |
| pair_010 | front | pair_010_front.mp4 | 16719807 | h264 | 3840x2160 | 60.0191 | 110 | 1.816089 | False | HIGH | 7.0 | 0.08 | 73d38d94138283cb… |
| pair_010 | back | pair_010_back.mp4 | 27153473 | hevc | 3840x2160 | 59.9295 | 256 | 4.255 | False | HIGH | 7.0 | 0.08 | be6abadb6bfe5e9c… |
| pair_011 | front | pair_011_front.mp4 | 36513305 | h264 | 3840x2160 | 60.0148 | 241 | 3.999011 | False | HIGH | 3.0 | -0.725 | d51a66a51291d49e… |
| pair_011 | back | pair_011_back.mp4 | 28975289 | hevc | 3840x2160 | 59.9344 | 275 | 4.571667 | False | HIGH | 3.0 | -0.725 | a550e83fba710b45… |
| pair_012 | front | pair_012_front.mp4 | 36373026 | h264 | 3840x2160 | 60.0164 | 238 | 3.948922 | False | HIGH | 3.0 | 1.395 | d0a18e872cb5e4f4… |
| pair_012 | back | pair_012_back.mp4 | 46923544 | hevc | 3840x2160 | 59.9327 | 446 | 7.425 | False | HIGH | 3.0 | 1.395 | 538aff0297038ea4… |
| pair_013 | front | pair_013_front.mp4 | pending transfer | h264 | 3840x2160 | 60.0161 | 404 | 6.714867 | False | HIGH | 3.0 | 6.695 | 9a5c1dae3e824495… |
| pair_013 | back | pair_013_back.mp4 | pending transfer | hevc | 3840x2160 | 40.1499 | 608 | 15.118333 | True | HIGH | 3.0 | 6.695 | 1f967dc25496f9ac… |
| pair_014 | front | pair_014_front.mp4 | 39585952 | h264 | 3840x2160 | 60.0152 | 266 | 4.415544 | False | HIGH | 3.0 | -1.82 | fcf62e9f10aef186… |
| pair_014 | back | pair_014_back.mp4 | 31434244 | hevc | 3840x2160 | 48.4724 | 276 | 5.673333 | True | HIGH | 3.0 | -1.82 | 682011e994bc0dd5… |
| pair_015 | front | pair_015_front.mp4 | 51136329 | h264 | 3840x2160 | 60.016 | 339 | 5.631833 | False | HIGH | 37.0 | -1.47 | c9e8d2e978dd9777… |
| pair_015 | back | pair_015_back.mp4 | 18422439 | hevc | 3840x2160 | 29.9683 | 190 | 6.306667 | False | HIGH | 37.0 | -1.47 | 8c52a8d0f56d037a… |
| pair_016 | front | pair_016_front.mp4 | pending transfer | h264 | 3840x2160 | 60.0152 | 551 | 9.164344 | False | HIGH | 10.0 | -1.675 | bb85759e452a2f30… |
| pair_016 | back | pair_016_back.mp4 | 30166035 | hevc | 3840x2160 | 29.9661 | 310 | 10.311667 | False | HIGH | 10.0 | -1.675 | 08cbd612e39d2feb… |
| pair_017 | front | pair_017_front.mp4 | 30325371 | h264 | 3840x2160 | 60.0165 | 200 | 3.315756 | False | HIGH | 3.0 | 0.355 | dc14bf8b16b700b7… |
| pair_017 | back | pair_017_back.mp4 | 20255650 | hevc | 3840x2160 | 29.9624 | 200 | 6.641667 | False | HIGH | 3.0 | 0.355 | 73017339d91a36e3… |
| pair_018 | front | pair_018_front.mp4 | 23436305 | h264 | 3840x2160 | 60.0205 | 155 | 2.565789 | False | HIGH | 3.0 | 1.405 | 2913662dc13a8f10… |
| pair_018 | back | pair_018_back.mp4 | 19376011 | hevc | 3840x2160 | 29.962 | 198 | 6.575 | False | HIGH | 3.0 | 1.405 | da908b0d4155357c… |
| pair_019 | front | pair_019_front.mp4 | pending transfer | h264 | 3840x2160 | 60.0152 | 372 | 6.181767 | False | MEDIUM | 8.0 | -2.38 | b4b46e750f94d9e0… |
| pair_019 | back | pair_019_back.mp4 | 25142817 | hevc | 3840x2160 | 59.9218 | 231 | 3.838333 | False | MEDIUM | 8.0 | -2.38 | e6ca02cccb2f914f… |
| pair_020 | front | pair_020_front.mp4 | 23615312 | h264 | 3840x2160 | 60.0148 | 158 | 2.616022 | False | HIGH | 3.0 | 3.32 | 0866ce06b40a4499… |
| pair_020 | back | pair_020_back.mp4 | 51935906 | hevc | 3840x2160 | 59.9265 | 490 | 8.16 | False | HIGH | 3.0 | 3.32 | f3178b178d8c1d5d… |
| pair_021 | front | pair_021_front.mp4 | pending transfer | h264 | 3840x2160 | 60.0152 | 348 | 5.781867 | False | HIGH | 3.0 | -1.25 | 9188530489f3954c… |
| pair_021 | back | pair_021_back.mp4 | 40694231 | hevc | 3840x2160 | 59.9372 | 383 | 6.373333 | False | HIGH | 3.0 | -1.25 | 4484c2b211c745f6… |
| pair_022 | front | pair_022_front.mp4 | 47354200 | h264 | 3840x2160 | 60.0151 | 312 | 5.182033 | False | MEDIUM | 3.0 | -1.565 | d09d89b6ba903c88… |
| pair_022 | back | pair_022_back.mp4 | 35142631 | hevc | 3840x2160 | 59.9276 | 332 | 5.523333 | False | MEDIUM | 3.0 | -1.565 | 2adc1123183b432f… |
| pair_023 | front | pair_023_front.mp4 | 48406490 | h264 | 3840x2160 | 60.0158 | 320 | 5.315267 | False | HIGH | 3.0 | -0.03 | 159b322e67dc1f46… |
| pair_023 | back | pair_023_back.mp4 | 45539616 | hevc | 3840x2160 | 59.9302 | 430 | 7.158333 | False | HIGH | 3.0 | -0.03 | 5d8b1282bd85afec… |
| pair_024 | front | pair_024_front.mp4 | pending transfer | h264 | 3840x2160 | 60.0153 | 481 | 7.997956 | False | HIGH | 3.0 | 1.235 | d99186aa3c77ab89… |
| pair_024 | back | pair_024_back.mp4 | pending transfer | hevc | 3840x2160 | 59.9266 | 736 | 12.265 | False | HIGH | 3.0 | 1.235 | 0f45265505e88997… |
| pair_025 | front | pair_025_front.mp4 | 37077497 | h264 | 3840x2160 | 60.015 | 244 | 4.048989 | False | HIGH | 3.0 | -0.59 | 5b9e78ccf24e6b10… |
| pair_025 | back | pair_025_back.mp4 | 35824844 | hevc | 3840x2160 | 59.9282 | 335 | 5.573333 | False | HIGH | 3.0 | -0.59 | 98442e7a9e5b1e2e… |

Full SHA-256 values are in `data/dual_camera_test_v1/metadata/pair_manifest.csv` (unchanged Phase 11 metadata). Frame counts and PTS statistics are from `results/phase12c_pts_sync/stream_pts_forensics_50_videos.csv` (Phase 12C).

## 4. Synchronization metadata (references — read-only, never modified)

- `data/dual_camera_test_v1/metadata/pair_manifest.csv` — per-pair files, durations, offsets, overlap, drift, residual, quality, SHA-256.
- `data/dual_camera_test_v1/metadata/synchronization_metadata.json` — 25 pair records (Phase 11.6).
- `data/dual_camera_test_v1/metadata/synchronization_quality.csv` — initial/final offsets, drift, drift status (STABLE_SUB_FRAME ×20, MINOR_DRIFT ×5).
- `data/dual_camera_test_v1/metadata/dataset_integrity_manifest.json`, `source_video_inventory.csv`, `pairing_analysis.csv`.
- `results/phase12c_pts_sync/pair_pts_alignment_validation_25_pairs.csv` + `phase12c_pts_validation_summary.json` — Phase 12C PTS validation: all 50 frame counts exact.
- Back-anchored reference clock with per-pair `global_offset_sec`; epsilon_sync (residual) is kept strictly separate from per-stream PTS discretization and from physical transit time, per Phase 12C.

## 5. Existing implementation used (approved phases — not rewritten)

- Phase 12C: `src/synchronization/pts_index.py`, `synchronized_reader.py` (sequential forward reading, PTS-indexed).
- Phase 12D: `src/detection/camera_detector.py` (COCO yolov8n: person/motorcycle only; Phase 9 best.pt: cranial helmet/no_helmet only), `src/tracking/camera_tracker.py`, `kalman_bytetrack.py`, `moto_rider_coupler.py`; config `configs/tracking_config.yaml`.
- Phase 12E: `src/association/*` (tracklet builder, candidate generator, scorer, global assignment, rider + orphan association, transit-time model); config `configs/cross_camera_association_config.yaml`.
- Phase 12E diagnostics previously ran on 4 pairs only (001, 013, 015, 019) as operational smoke tests — **no annotation dataset exists yet**: `data/dual_camera_test_v1/annotations/` does not exist in the repository (verified: dataset folder contains only back/, front/, metadata/, previews/, unmatched_quarantine/).
- Known detector limitation (Phase 12D benchmark, 30-tick samples): back-camera COCO motorcycle detections = 0 and helmet-model head detections = 0 on several back videos (rear views). Back-camera annotation therefore requires visual verification in the audit passes; detector silence is never treated as absence.

## 6. Models (hash-verified on the VM)

| Model | Path | SHA-256 | Classes |
|---|---|---|---|
| Phase 9 helmet (seed 42, EXP-COMBINED) | `results/phase9_multiseed/exp_combined/seed_42/weights/best.pt` | `0929a882667e0f7e2b631071ab218a2415cc1b816cdede7fb6812543d840b0e5` ✓ matches `configs/model_registry.yaml` + `configs/tracking_config.yaml` | 0=no_helmet, 1=helmet (cranial) |
| COCO | `yolov8n.pt` | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` ✓ matches `configs/tracking_config.yaml` | 0=person, 3=motorcycle used |

## 7. Compute resources

- VM (primary): 4 CPU cores, 7.7 GB RAM, 108 GB free disk, ffmpeg/ffprobe 8.1.2, Python 3.12 venv with torch (CPU), ultralytics, opencv. Videos total ~2.0 GB; annotation outputs are text-scale.
- Mac (source only): M1 MacBook Air; no processing is run on the Mac in this mission.
- Kaggle T4: account elafahmed6563 (verified in earlier projects; token kept in the shared `~/.kaggle` — a project-scoped config will be used and the account re-verified before any use). GPU will be used only if the local CPU benchmark shows it materially improves throughput; quota will be checked first and not exhausted.

## 8. Problems discovered

1. **7 videos pending transfer** (size limit) — see §2. Pairs 013, 016, 019, 021, 024 cannot start until resolved; all other 18 pairs are fully ready (both videos hash-verified).
2. **Back-camera detector blindness** (Phase 12D evidence) — mitigated by mandatory visual audit of back views; missed-object corrections will carry `MANUAL_CORRECTION` provenance.
3. **Scientific-integrity tension:** the Phase 12B protocol prohibits unverified model-bootstrapped ground truth and requires human visual verification. This mission is executed as AI-assisted annotation with AI re-audits under the mission's explicit instruction; the product is therefore labelled `AI_ASSISTED_AUDITED` and must not be cited as human-verified ground truth. This limitation is carried into the final report.
4. `git clone` of the repository fails on this network (fetch-pack early EOF); the codeload tarball of the same commit was used for the VM mirror and its content was spot-verified against the Mac (docs, configs, manifests identical).

