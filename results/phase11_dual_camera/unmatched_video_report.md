# UNMATCHED VIDEO QUARANTINE REPORT — PHASE 11

**Project:** Dual-Camera Context-Aware Motorcycle Helmet Violation Detection  
**Module:** Phase 11 — Dual-Camera Video Synchronization & Benchmark Preparation  
**Date:** November 7, 2026  
**Auditor:** Senior Computer Vision Research Scientist & Data Quality Auditor  
**Status:** 2 Unmatched Files Quarantined & Cryptographically Preserved  

---

## 1. Executive Summary

Of the 27 original Front-camera video recordings provided in `Motorcycle synchronization/Front/`, exactly **25 videos** formed confirmed, 1-to-1 synchronized pairs with the 25 Back-camera recordings (`IMG_0620.MOV` through `IMG_0644.MOV`).

The remaining **2 Front-camera recordings** have been isolated and preserved in:
`data/dual_camera_test_v1/unmatched_quarantine/`

In strict accordance with the project's **Data-Preservation Rules**, these videos were **NOT deleted, truncated, or modified**. They are held in quarantine with full cryptographic SHA-256 provenance for transparent academic auditing.

---

## 2. Quarantined Videos Inventory

| Quarantined Filename | Original Filename | File Size | Duration | Resolution | FPS | Creation Timestamp (UTC) | SHA-256 Checksum | Quarantine Classification |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `unmatched_01_front_11.mp4` | `11.mp4` | 14.5 MB | 1.78s | 3840x2160 | 60.02 | 2026-10-10 07:32:54Z | `a59c4690c7c34b6b64703a812e98fa6a836da10c85b5961d2d3127609204c311` | Secondary Overtaking Sub-Clip |
| `unmatched_02_front_16.mp4` | `16.mp4` | 18.2 MB | 2.12s | 3840x2160 | 60.02 | 2026-10-10 07:36:46Z | `94073beb61de1f84b6f007e0c8bdf71ea8d1d81f2fc50058e1762c3e17cf6d64` | Post-Event Trailing Clip |

---

## 3. Detailed Forensic & Visual Matching Analysis

### 3.1 Case 1: `11.mp4` (Original Source: `Front/11.mp4`)

#### Context & Timeline:
- **Back Video Candidate:** `IMG_0629.MOV` (Recorded: 07:32:49Z to 07:32:53.27Z, Duration: $4.27\text{s}$)
- **Front Video A:** `10.mp4` (Recorded: 07:32:51Z, Duration: $1.90\text{s}$)
- **Front Video B:** `11.mp4` (Recorded: 07:32:54Z, Duration: $1.78\text{s}$)

#### Visual & Acoustic Findings:
1. **Multi-Rider Overtaking Event:** Inspection of `IMG_0629.MOV` reveals two motorcycles traveling closely in succession:
   - Lead motorcycle: Rider in a yellow t-shirt with a shoulder courier bag and red/black helmet.
   - Trailing motorcycle: Rider in a black t-shirt and dark helmet on a red/black commuter motorcycle.
2. **Operator Behavior:** The front-camera operator did not record a single continuous video. Instead, the operator pressed stop and start rapidly:
   - `10.mp4` captured the front view of the lead yellow-shirt rider passing by.
   - `11.mp4` captured the front view of the second trailing rider.
3. **Pair Assignment Rationale:**
   - `10.mp4` overlaps directly with the primary temporal segment of `IMG_0629.MOV` (Audio correlation peak $R = 0.3098$, offset $+0.080\text{s}$). It is designated as the primary partner for `pair_010`.
   - `11.mp4` started at 07:32:54Z, as `IMG_0629.MOV` was cutting off. While it depicts the second rider from the same traffic scene, treating it as an independent pair would violate 1-to-1 temporal correspondence since no second back video exists.
   - **Verdict:** Quarantined as `unmatched_01_front_11.mp4`.

---

### 3.2 Case 2: `16.mp4` (Original Source: `Front/16.mp4`)

#### Context & Timeline:
- **Back Video Candidate:** `IMG_0633.MOV` (Recorded: 07:36:38Z to 07:36:43.69Z, Duration: $5.69\text{s}$)
- **Front Video A:** `15.mp4` (Recorded: 07:36:40Z, Duration: $4.48\text{s}$)
- **Front Video B:** `16.mp4` (Recorded: 07:36:46Z, Duration: $2.12\text{s}$)

#### Visual & Acoustic Findings:
1. **Subject Identification:**
   - In `IMG_0633.MOV`, the back camera captures a rider riding away wearing a black sports jersey with yellow lettering "**ARFAN 40**" on the back and NO HELMET.
   - In `15.mp4`, the front camera captures the exact same rider approaching head-on with bare head and dark jersey with yellow trim.
   - Acoustic cross-correlation between `IMG_0633.MOV` and `15.mp4` is exceptionally strong ($R = 0.3600$, overlap $2.66\text{s}$).
2. **Post-Event Video Disconnect:**
   - `16.mp4` was recorded at 07:36:46Z—more than **2.3 seconds after `IMG_0633.MOV` stopped recording**.
   - Visual inspection of `16.mp4` shows the front camera operator panned backwards toward the junction where a yellow motorcycle is turning, while rider #40 is already a tiny speck in the far distance.
   - Acoustic correlation with `IMG_0633.MOV` is weak ($R = 0.2172$), with no concurrent motorcycle pass-by sound.
3. **Pair Assignment Rationale:**
   - `15.mp4` is the unambiguous true front-camera recording of rider #40 for `pair_014`.
   - `16.mp4` is an extra trailing recording captured after the target vehicle had cleared the camera baseline.
   - **Verdict:** Quarantined as `unmatched_02_front_16.mp4`.

---

## 4. Preservation & Re-integration Protocol

If future multi-view tracking research specifically investigates multi-target overtaking or multi-clip concatenation, both quarantined videos can be referenced via their manifest metadata in `data/dual_camera_test_v1/metadata/dataset_integrity_manifest.json`.

For the primary 25-pair synchronous benchmark (`data/dual_camera_test_v1/`), quarantine isolation ensures clean 1-to-1 temporal correspondence with zero ambiguity.

---
*Report certified by Independent Research Data Quality Auditor.*
