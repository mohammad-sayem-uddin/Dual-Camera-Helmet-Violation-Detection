# Forensic Audit & Initial Dataset Inventory Report

**Document**: `01_initial_dataset_inventory.md`  
**Dataset Under Audit**: `Motorcyle dataset`  
**Auditor**: Senior Computer Vision Research Engineer & Dataset Curator  
**Timestamp**: Tue Nov  3 20:12:12 UTC 2026  
**Status**: Completed & Ingested into Isolated Expansion Workspace  

---

## 1. Executive Summary

A comprehensive forensic audit and byte-for-byte ingestion was conducted on the newly provided dataset folder `Motorcyle dataset`. 
All images have been verified for file integrity, image decoding, exact cryptographic duplication, perceptual near-duplication, and sequence affinity.
All 1,841 original source images have been preserved byte-for-byte in `data/helmet_dataset_expansion/original/` with verified cryptographic SHA-256 match.

| Forensic Metric | Value | Verification Status |
| :--- | :---: | :--- |
| **Total Images Identified** | **1,841** | Confirmed (100% matched expected count) |
| **Readable / Decodable Images** | **1841** | 100% verified via PIL byte decoder |
| **Corrupted / Truncated Files** | **0** | Zero corruption detected |
| **Blank / Solid Color Images** | **0** | Zero blank/solid images |
| **Cryptographic Exact Duplicate Groups** | **0** | 0 duplicate files identified |
| **Perceptual Near-Duplicate Pairs (Hamming $\le 4$)** | **0** | Clustered for sequence isolation |
| **Unique Semantic / Sequence Groups** | **565** | Extracted from filename naming schemes |
| **Byte-for-Byte Preservation** | **100.0%** | All 1,841 SHA-256 hashes verified in destination |

---

## 2. Image Resolution & Format Profile

All files are standard JPEG images (`.jpg`).

### Image Dimensions Distribution (Top 10):
| Width $\times$ Height | Image Count | Percentage |
| :---: | :---: | :---: |
| 1024 $\times$ 1024 | 1841 | 100.00% |

### Summary Statistics:
- **Min Width / Height**: 1024 px / 1024 px
- **Max Width / Height**: 1024 px / 1024 px
- **Mean Aspect Ratio**: 1.000 (Std: 0.000)
- **Min File Size**: 610.1 KB
- **Max File Size**: 1267.5 KB
- **Mean File Size**: 1022.6 KB

---

## 3. Cryptographic Duplicate & Near-Duplicate Analysis

### 3.1 Exact Cryptographic Duplicates (Identical SHA-256)
- **Total Duplicate Groups**: 0
- **Redundant Files**: 0

No exact cryptographic duplicates were found in the dataset.

### 3.2 Perceptual Near-Duplicates (64-bit dHash Hamming Distance $\le 4$)
- Total near-duplicate candidate pairs identified: **0**
- These pairs represent adjacent video frames, slightly cropped variations, or rapid continuous burst shots.
- **Leakage Prevention Directive**: As mandated by Section 12, all near-duplicate candidate pairs and semantic group members will be assigned to identical splits (never split across train/val/test).

---

## 4. Sequence & Semantic Group Extraction

Analysis of the filename naming conventions revealed **565 semantic sequence groups**.
Top groups include:
- `Rider_without_helmet_on_motorcycle`: 102 images
- `Motorcycle_rider_wearing_helmet`: 86 images
- `Motorcycle_rider_without_helmet`: 84 images
- `Motorcyclist_riding_without_helmet`: 68 images
- `Two_riders_without_helmets`: 55 images
- `Two_riders_wearing_helmets`: 36 images
- `Two_riders_with_mixed_helmets`: 35 images
- `Rider_without_helmet_on_bike`: 29 images
- `Rider_wearing_helmet_on_motorcycle`: 28 images
- `Motorcycle_riders_with_mixed_hel…`: 27 images
- `Two_riders_on_motorcycles`: 27 images
- `Rider_wearing_helmet_on_bike`: 23 images
- `Motorcycles_riding_on_road`: 21 images
- `Two_riders_on_motorcycle`: 17 images
- `Rider_without_helmet_on_street`: 16 images

---

## 5. Visual Artifacts

The distribution profile has been saved to:
- `data/helmet_dataset_expansion/visualizations/initial_inventory_characteristics.png`

---

## 6. Next Steps

With forensic ingestion complete and original images frozen byte-for-byte, execution proceeds immediately to:
1. Grounding annotation guidelines (Class 0: `no_helmet`, Class 1: `helmet`, tight head/helmet bounding boxes).
2. Autonomous annotation across all 1,841 images using the multimodal vision pipeline.
3. Conducting Audit Passes 1, 2, and 3.
