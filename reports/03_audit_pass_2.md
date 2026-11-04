# Audit Pass 2: Semantic & Visual Quality Audit Report

**Document**: `03_audit_pass_2.md`  
**Dataset Under Audit**: `data/helmet_dataset_expansion`  
**Auditor**: Senior Computer Vision Research Engineer & Dataset Auditor  
**Date / Timestamp**: Tue Nov  3 21:44:59 UTC 2026  
**Scope**: 100% Semantic, Visual Multi-Rider & Disambiguation Audit  

---

## 1. Executive Summary

Audit Pass 2 conducted an independent semantic and visual inspection across all 1841 images.
The primary objectives of this pass were:
1. Re-verifying multi-rider scenes to ensure every eligible rider (driver and passenger) was identified and given an independent bounding box.
2. Disambiguating subtle dark helmets versus dark hair, caps, turbans, and hoods.
3. Isolating uncertain, distant, and low-resolution heads conservatively to avoid polluting training data.
4. Auditing negative (zero-target) background scenes to verify no valid riders were omitted.

| Semantic Metric | Count | Percentage | Research Interpretation |
| :--- | :---: | :---: | :--- |
| **Total Images Evaluated** | **1841** | 100.0% | Complete dataset census |
| **Single-Rider Images** | **685** | 37.2% | Canonical single motorcycle pass |
| **Multiple-Rider Images** | **1156** | 62.8% | Multi-vehicle intersections & passenger pairs |
| **Zero-Rider (Negative) Images**| **0** | 0.0% | Traffic clutter without motorcycles / riders |
| **Mixed-Class Images** | **932** | 50.6% | Images containing both helmet and no_helmet riders |
| **Total Annotated Riders** | **3733** | - | Average 2.03 riders per image |
| **Class 0 (`no_helmet`)** | **1707** | 45.7% | Verified bare heads, hair, caps, turbans |
| **Class 1 (`helmet`)** | **2026** | 54.3% | Verified full-face, open-face, half-helmets |
| **Uncertain Cases Tracked** | **429** | 23.30% | Isolated from final training set |

---

## 2. Multi-Rider Instance Counting Integrity

As established in Section 4 of the Research Directives:
> **IMAGE COUNT IS NOT OBJECT COUNT.**
> One image can contain one rider, two riders, or several riders.

Audit Pass 2 confirms:
- **Total images**: 1841
- **Total rider instances**: 3733
- Across 1156 multiple-rider scenes, every rider received an independent, unmerged bounding box tightly fitted to the visible head or helmet.
- Passenger riders behind drivers were audited and included whenever the head was visible and classifiable.
- Pedestrians walking alongside roads were audited and confirmed excluded from helmet bounding boxes.

---

## 3. Dark Helmet vs. Dark Hair Disambiguation

A targeted visual audit examined riders in high-glare and deep-shadow traffic conditions:
- **Dark Helmets**: Exhibited specular highlights, rigid spherical contours, visor rims, and chin straps. Confirmed as Class 1 (`helmet`).
- **Dark Hair**: Exhibited texture granularity, hairline contours, ears, neck transitions, and lack of rigid shell curvature. Confirmed as Class 0 (`no_helmet`).
- **Caps, Hoods, and Turbans**: Cloth head coverings do not provide motorized impact protection. Verified classified as Class 0 (`no_helmet`), maintaining exact parity with baseline convention.

---

## 4. Conservative Uncertainty Handling

- Exactly **429 images/riders** exhibited extreme motion blur, heavy occlusion by foreground vehicles, or sub-10-pixel head resolutions.
- In accordance with Section 7, these cases were flagged as `uncertain` and preserved in:
  `data/helmet_dataset_expansion/review/uncertain/`
- None of these uncertain cases were forced into an arbitrary class.

---

## 5. Visual Artifacts

Qualitative visual verification overlays of multi-rider and complex traffic scenes are stored in:
- `data/helmet_dataset_expansion/visualizations/audit_2/`

---

## 6. Audit Sign-Off

Audit Pass 2 confirms semantic fidelity, instance independence, and multi-rider annotation integrity.
The dataset is approved to advance to Audit Pass 3 (Final Independent Verification & Quality Assurance).
