# PHASE 11A: SOURCE VIDEO INVENTORY & DATASET INTEGRITY REPORT

**Project:** Dual-Camera Context-Aware Motorcycle Helmet Violation Detection  
**Module:** Phase 11 — Dual-Camera Video Synchronization & Benchmark Preparation  
**Date:** November 7, 2026  
**Auditor:** Senior Computer Vision Research Scientist & Data Quality Auditor  
**Status:** Audit Complete — 52 Files Inspected (25 Back, 27 Front)  

---

## 1. Executive Summary & Inventory Overview

A comprehensive, non-destructive audit of the raw dual-camera video corpus provided under `Motorcycle synchronization/` was conducted. All original files were accessed in read-only mode, and cryptographic SHA-256 digests were computed to guarantee data provenance and reproducibility.

### Summary Metrics:
- **Total Source Videos Inspected:** 52
- **Back-Camera Videos:** 25 (`IMG_0620.MOV` through `IMG_0644.MOV`)
- **Front-Camera Videos:** 27 (`1.mp4` through `27.mp4`)
- **Expected Synchronized Pairs:** 25
- **Excess Unpaired Front Videos:** 2 (`10.mp4` / `11.mp4` cluster and `15.mp4` / `16.mp4` cluster)
- **Decodability & Health Status:** 52/52 ($100\%$) videos fully decodable with valid video and audio streams
- **Corrupted / Zero-Byte Files:** 0
- **Total Data Volume:** ~1.85 GB

---

## 2. Technical Recording Specifications

| Stream / Characteristic | Back-Camera Stream (`Back/`) | Front-Camera Stream (`Front/`) | Compatibility Notes |
| :--- | :--- | :--- | :--- |
| **Recording Device** | Apple iPad (10th Generation) | Android 13 Smartphone | Heterogeneous mobile hardware |
| **Container Format** | QuickTime / MOV (`.MOV`) | ISO MP4 / AVC (`.mp4`) | Containers differ; both standards-compliant |
| **Video Codec** | HEVC / H.265 (`hvc1`) | H.264 / AVC (`avc1`) | Both natively decodable via FFmpeg / PyAV |
| **Spatial Resolution** | $3840 \times 2160$ (4K UHD) | $3840 \times 2160$ (4K UHD) | **Identical 4K Ultra-HD native resolution** |
| **Aspect Ratio** | 16:9 widescreen | 16:9 widescreen | Perfectly matched spatial geometry |
| **Frame Rate** | ~59.93 fps nominal ($29.96$--$59.94$ observed) | ~60.015 fps nominal | Both nominal 60 fps; minor VFR variations |
| **Audio Stream** | AAC-LC ($48,000\text{ Hz}$, Mono) | AAC-LC ($48,000\text{ Hz}$, Stereo) | **Matching 48 kHz sampling rate for cross-correlation** |
| **Recording Session** | 2026-10-10 07:25:53Z to 07:48:20Z | 2026-10-10 07:25:56Z to 07:48:23Z | Identical ~23-minute roadside recording session |

---

## 3. Back-Camera Inventory Table (25 Recordings)

| Filename | File Size | Duration | Resolution | FPS | Video Codec | Audio Codec (Rate) | Creation Timestamp (UTC) | SHA-256 (Truncated) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| `IMG_0620.MOV` | 40.0 MB | 6.62s | 3840x2160 | 59.93 | hevc | aac (48k mono) | 2026-10-10 07:25:53Z | `da982b7b94c7...` | OK |
| `IMG_0621.MOV` | 42.5 MB | 7.06s | 3840x2160 | 59.93 | hevc | aac (48k mono) | 2026-10-10 07:27:03Z | `31876bf9ed0a...` | OK |
| `IMG_0622.MOV` | 44.0 MB | 7.28s | 3840x2160 | 59.93 | hevc | aac (48k mono) | 2026-10-10 07:27:31Z | `c04f7865d5fa...` | OK |
| `IMG_0623.MOV` | 20.4 MB | 3.35s | 3840x2160 | 59.94 | hevc | aac (48k mono) | 2026-10-10 07:28:42Z | `1437062ad662...` | OK |
| `IMG_0624.MOV` | 35.3 MB | 5.87s | 3840x2160 | 59.93 | hevc | aac (48k mono) | 2026-10-10 07:29:02Z | `3288849c5269...` | OK |
| `IMG_0625.MOV` | 25.9 MB | 4.25s | 3840x2160 | 59.93 | hevc | aac (48k mono) | 2026-10-10 07:29:31Z | `da452a266aac...` | OK |
| `IMG_0626.MOV` | 24.4 MB | 3.99s | 3840x2160 | 59.93 | hevc | aac (48k mono) | 2026-10-10 07:29:46Z | `737a50b5785a...` | OK |
| `IMG_0627.MOV` | 27.7 MB | 4.50s | 3840x2160 | 59.93 | hevc | aac (48k mono) | 2026-10-10 07:31:56Z | `e1862359cfc6...` | OK |
| `IMG_0628.MOV` | 42.0 MB | 6.92s | 3840x2160 | 59.93 | hevc | aac (48k mono) | 2026-10-10 07:32:14Z | `8aed96e1bd2c...` | OK |
| `IMG_0629.MOV` | 25.9 MB | 4.27s | 3840x2160 | 59.93 | hevc | aac (48k mono) | 2026-10-10 07:32:49Z | `932d6d772782...` | OK |
| `IMG_0630.MOV` | 27.7 MB | 4.59s | 3840x2160 | 59.93 | hevc | aac (48k mono) | 2026-10-10 07:33:07Z | `146d65370eef...` | OK |
| `IMG_0631.MOV` | 44.8 MB | 7.44s | 3840x2160 | 59.93 | hevc | aac (48k mono) | 2026-10-10 07:33:25Z | `486f6f419cd2...` | OK |
| `IMG_0632.MOV` | 60.9 MB | 15.13s | 3840x2160 | 40.17 | hevc | aac (48k mono) | 2026-10-10 07:35:21Z | `98f99e14f7c3...` | OK |
| `IMG_0633.MOV` | 30.0 MB | 5.69s | 3840x2160 | 48.51 | hevc | aac (48k mono) | 2026-10-10 07:36:38Z | `a643ed07d652...` | OK |
| `IMG_0634.MOV` | 17.6 MB | 6.34s | 3840x2160 | 29.97 | hevc | aac (48k mono) | 2026-10-10 07:37:45Z | `56fdaf7ccf31...` | OK |
| `IMG_0635.MOV` | 28.8 MB | 10.33s | 3840x2160 | 30.02 | hevc | aac (48k mono) | 2026-10-10 07:38:04Z | `24df519c2835...` | OK |
| `IMG_0636.MOV` | 19.4 MB | 6.66s | 3840x2160 | 30.04 | hevc | aac (48k mono) | 2026-10-10 07:40:23Z | `50e13128efb6...` | OK |
| `IMG_0637.MOV` | 18.5 MB | 6.61s | 3840x2160 | 29.96 | hevc | aac (48k mono) | 2026-10-10 07:40:44Z | `0c46f17a8ef4...` | OK |
| `IMG_0638.MOV` | 24.0 MB | 3.85s | 3840x2160 | 59.92 | hevc | aac (48k mono) | 2026-10-10 07:41:53Z | `d31ac94d430c...` | OK |
| `IMG_0639.MOV` | 49.6 MB | 8.18s | 3840x2160 | 59.93 | hevc | aac (48k mono) | 2026-10-10 07:43:37Z | `ea8cb3dd45d1...` | OK |
| `IMG_0640.MOV` | 38.9 MB | 6.39s | 3840x2160 | 59.94 | hevc | aac (48k mono) | 2026-10-10 07:43:58Z | `33218d5016de...` | OK |
| `IMG_0641.MOV` | 33.6 MB | 5.54s | 3840x2160 | 59.93 | hevc | aac (48k mono) | 2026-10-10 07:45:17Z | `b6c4981df5d7...` | OK |
| `IMG_0642.MOV` | 43.5 MB | 7.17s | 3840x2160 | 59.93 | hevc | aac (48k mono) | 2026-10-10 07:45:32Z | `d0dcf7db1c6c...` | OK |
| `IMG_0643.MOV` | 73.5 MB | 12.28s | 3840x2160 | 59.93 | hevc | aac (48k mono) | 2026-10-10 07:47:54Z | `382f732d7dd3...` | OK |
| `IMG_0644.MOV` | 34.2 MB | 5.59s | 3840x2160 | 59.93 | hevc | aac (48k mono) | 2026-10-10 07:48:20Z | `8c29fd7e5256...` | OK |

---

## 4. Front-Camera Inventory Table (27 Recordings)

| Filename | File Size | Duration | Resolution | FPS | Video Codec | Audio Codec (Rate) | Creation Timestamp (UTC) | SHA-256 (Truncated) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| `1.mp4` | 30.1 MB | 3.45s | 3840x2160 | 59.73 | h264 | aac (48k stereo) | 2026-10-10 07:25:56Z | `995d1fa69ea4...` | OK |
| `2.mp4` | 45.0 MB | 5.28s | 3840x2160 | 60.02 | h264 | aac (48k stereo) | 2026-10-10 07:27:08Z | `b902c441e109...` | OK |
| `3.mp4` | 33.7 MB | 3.96s | 3840x2160 | 60.02 | h264 | aac (48k stereo) | 2026-10-10 07:27:37Z | `17f2ace3dccb...` | OK |
| `4.mp4` | 25.7 MB | 3.06s | 3840x2160 | 60.02 | h264 | aac (48k stereo) | 2026-10-10 07:28:43Z | `dff63a5e00eb...` | OK |
| `5.mp4` | 29.7 MB | 3.48s | 3840x2160 | 60.02 | h264 | aac (48k stereo) | 2026-10-10 07:29:05Z | `0c041d229095...` | OK |
| `6.mp4` | 19.2 MB | 2.30s | 3840x2160 | 60.02 | h264 | aac (48k stereo) | 2026-10-10 07:29:32Z | `758a97a08e59...` | OK |
| `7.mp4` | 22.6 MB | 2.67s | 3840x2160 | 60.02 | h264 | aac (48k stereo) | 2026-10-10 07:29:48Z | `20177584b3b6...` | OK |
| `8.mp4` | 22.7 MB | 2.66s | 3840x2160 | 60.01 | h264 | aac (48k stereo) | 2026-10-10 07:31:58Z | `a09374101a00...` | OK |
| `9.mp4` | 33.5 MB | 3.92s | 3840x2160 | 60.02 | h264 | aac (48k stereo) | 2026-10-10 07:32:18Z | `b49f5204a5ef...` | OK |
| `10.mp4` | 15.9 MB | 1.90s | 3840x2160 | 60.02 | h264 | aac (48k stereo) | 2026-10-10 07:32:51Z | `73d38d941382...` | Candidate A |
| `11.mp4` | 14.5 MB | 1.78s | 3840x2160 | 60.02 | h264 | aac (48k stereo) | 2026-10-10 07:32:54Z | `a59c4690c7c3...` | Candidate B |
| `12.mp4` | 34.8 MB | 4.08s | 3840x2160 | 60.02 | h264 | aac (48k stereo) | 2026-10-10 07:33:10Z | `d51a66a51291...` | OK |
| `13.mp4` | 34.7 MB | 4.03s | 3840x2160 | 60.02 | h264 | aac (48k stereo) | 2026-10-10 07:33:28Z | `d0a18e872cb5...` | OK |
| `14.mp4` | 57.8 MB | 6.78s | 3840x2160 | 60.02 | h264 | aac (48k stereo) | 2026-10-10 07:35:33Z | `9a5c1dae3e82...` | OK |
| `15.mp4` | 37.8 MB | 4.48s | 3840x2160 | 60.02 | h264 | aac (48k stereo) | 2026-10-10 07:36:40Z | `fcf62e9f10ae...` | Candidate A |
| `16.mp4` | 18.2 MB | 2.12s | 3840x2160 | 60.02 | h264 | aac (48k stereo) | 2026-10-10 07:36:46Z | `94073beb61de...` | Candidate B |
| `17.mp4` | 48.8 MB | 5.72s | 3840x2160 | 60.02 | h264 | aac (48k stereo) | 2026-10-10 07:37:48Z | `c9e8d2e978dd...` | OK |
| `18.mp4` | 78.8 MB | 9.26s | 3840x2160 | 60.02 | h264 | aac (48k stereo) | 2026-10-10 07:38:12Z | `bb85759e452a...` | OK |
| `19.mp4` | 28.9 MB | 3.41s | 3840x2160 | 60.02 | h264 | aac (48k stereo) | 2026-10-10 07:40:26Z | `dc14bf8b16b7...` | OK |
| `20.mp4` | 22.4 MB | 2.64s | 3840x2160 | 60.02 | h264 | aac (48k stereo) | 2026-10-10 07:40:47Z | `2913662dc13a...` | OK |
| `21.mp4` | 54.1 MB | 6.28s | 3840x2160 | 60.02 | h264 | aac (48k stereo) | 2026-10-10 07:41:57Z | `b4b46e750f94...` | OK |
| `22.mp4` | 22.5 MB | 2.70s | 3840x2160 | 60.02 | h264 | aac (48k stereo) | 2026-10-10 07:43:43Z | `0866ce06b40a...` | OK |
| `23.mp4` | 50.2 MB | 5.84s | 3840x2160 | 60.02 | h264 | aac (48k stereo) | 2026-10-10 07:44:02Z | `9188530489f3...` | OK |
| `24.mp4` | 45.2 MB | 5.25s | 3840x2160 | 60.02 | h264 | aac (48k stereo) | 2026-10-10 07:45:20Z | `d09d89b6ba90...` | OK |
| `25.mp4` | 46.2 MB | 5.41s | 3840x2160 | 60.02 | h264 | aac (48k stereo) | 2026-10-10 07:45:36Z | `159b322e67dc...` | OK |
| `26.mp4` | 70.6 MB | 8.05s | 3840x2160 | 60.02 | h264 | aac (48k stereo) | 2026-10-10 07:48:02Z | `d99186aa3c77...` | OK |
| `27.mp4` | 35.4 MB | 4.12s | 3840x2160 | 60.02 | h264 | aac (48k stereo) | 2026-10-10 07:48:23Z | `5b9e78ccf24e...` | OK |

---

## 5. Preliminary Observations for Phase 11B Matching

1. **Systematic Timestamp Correlation**:
   The front camera was started consistently $1$ to $5$ seconds after the back camera for almost every recording sequence. The back operator initiated recording as the motorcycle approached, and the front operator initiated recording as the vehicle passed into the front view.
2. **Ambiguity Isolation**:
   - **Cluster 1**: Back `IMG_0629.MOV` (07:32:49Z, $4.27\text{s}$) corresponds to either Front `10.mp4` (07:32:51Z, $1.90\text{s}$) or `11.mp4` (07:32:54Z, $1.78\text{s}$). One is a false start or duplicate attempt; one is the true synchronized partner.
   - **Cluster 2**: Back `IMG_0633.MOV` (07:36:38Z, $5.69\text{s}$) corresponds to either Front `15.mp4` (07:36:40Z, $4.48\text{s}$) or `16.mp4` (07:36:46Z, $2.12\text{s}$).
3. **Audio Cross-Correlation Viability**:
   Both cameras captured high-fidelity AAC audio at $48\text{ kHz}$. Audio cross-correlation between engine rumble, horn honks, and roadside ambient acoustic spikes provides an objective, quantitative ground-truth signal for measuring sub-frame temporal offset and resolving candidate ambiguities.

---
*Report certified by Independent Research Data Auditor. Raw manifest saved to `source_video_inventory.csv`.*
