# HUMAN VERIFICATION GUIDE — PHASE 11 SYNCHRONIZED DUAL-CAMERA BENCHMARK

**Project:** Dual-Camera Context-Aware Motorcycle Helmet Violation Detection  
**Module:** Phase 11 — Dual-Camera Video Synchronization & Benchmark Preparation  
**Target Audience:** Lead CV Researcher & Human Evaluation Auditor  
**Date:** November 7, 2026  

---

## 1. Purpose of Human Review

This verification package enables the researcher to systematically inspect and validate all **25 synchronized front/back video pairs** before proceeding to cross-camera detection, homography projection, or evidence fusion.

Although automated audio cross-correlation and metadata alignment have achieved **100% verification with 0 failed pairs**, visual confirmation by the domain expert ensures that:
1. Every pair captures the **exact same motorcycle and rider(s)**.
2. The synchronized frame previews reflect true temporal concurrence.
3. The 3 pairs marked with `MEDIUM` quality acoustic correlation are visually verified.
4. The 2 quarantined recordings (`11.mp4` and `16.mp4`) are confirmed as non-primary clips.

---

## 2. Preview Package Location & Structure

High-resolution ($1920 \times 600$) synchronized side-by-side composite images have been pre-rendered for all 25 pairs:

- **Centralized Previews Directory:**  
  [`results/phase11_dual_camera/synchronized_preview/`](file:///Users/sayemuddin/Desktop/research_helmet/results/phase11_dual_camera/synchronized_preview/)
- **Dataset Structure Previews:**  
  [`data/dual_camera_test_v1/previews/pair_XXX/`](file:///Users/sayemuddin/Desktop/research_helmet/data/dual_camera_test_v1/previews/)

### Composite Format:
- **Left Panel:** Back camera frame at midpoint of overlap ($t_{\text{back}}$).
- **Right Panel:** Front camera frame at synchronized timestamp ($t_{\text{front}} = t_{\text{back}} - t_{\text{offset}}$).
- **Top Banner:** Pair ID, source filenames, extracted timestamps, and measured temporal offset.

---

## 3. Comprehensive Pair-by-Pair Review Guide

| Pair ID | Back Source | Front Source | Offset ($t_{\text{offset}}$) | Overlap Duration | Quality | Visual Matching Anchor & Key Landmarks | Researcher Inspection Checklist |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **`pair_001`** | `IMG_0620.MOV` | `1.mp4` | $-0.205\text{s}$ | $3.10\text{s}$ | **HIGH** | Maroon shirt rider on black motorcycle (NO HELMET); blue van in back view | [ ] Verified |
| **`pair_002`** | `IMG_0621.MOV` | `2.mp4` | $-1.055\text{s}$ | $4.21\text{s}$ | **HIGH** | Commuter motorcycle; dirt road junction; pedestrian walking past pole | [ ] Verified |
| **`pair_003`** | `IMG_0622.MOV` | `3.mp4` | $+0.665\text{s}$ | $3.95\text{s}$ | **HIGH** | Approaching rider; building construction visible in background field | [ ] Verified |
| **`pair_004`** | `IMG_0623.MOV` | `4.mp4` | $-1.285\text{s}$ | $1.77\text{s}$ | **HIGH** | Fast pass-by; high audio engine rev spike; dirt track curvature | [ ] Verified |
| **`pair_005`** | `IMG_0624.MOV` | `5.mp4` | $+0.540\text{s}$ | $3.48\text{s}$ | **HIGH** | Distinct helmet visor reflection; brick edging on roadside | [ ] Verified |
| **`pair_006`** | `IMG_0625.MOV` | `6.mp4` | $-0.045\text{s}$ | $2.24\text{s}$ | **HIGH** | Near-zero offset ($-45\text{ ms}$); rider in dark attire | [ ] Verified |
| **`pair_007`** | `IMG_0626.MOV` | `7.mp4` | $+0.415\text{s}$ | $2.65\text{s}$ | **HIGH** | Clean acoustic peak ($R = 0.5435$); matching utility pole posters | [ ] Verified |
| **`pair_008`** | `IMG_0627.MOV` | `8.mp4` | $+0.755\text{s}$ | $2.65\text{s}$ | **MEDIUM** | Moderate wind noise; check rider jacket color and motorcycle headlight | [ ] **Inspect Medium** |
| **`pair_009`** | `IMG_0628.MOV` | `9.mp4` | $+0.225\text{s}$ | $3.90\text{s}$ | **HIGH** | Clean correlation ($R = 0.4919$); road junction corner | [ ] Verified |
| **`pair_010`** | `IMG_0629.MOV` | `10.mp4` | $+0.080\text{s}$ | $1.90\text{s}$ | **HIGH** | Yellow-shirt delivery rider with side courier bag and red/black helmet | [ ] Verified |
| **`pair_011`** | `IMG_0630.MOV` | `12.mp4` | $-0.725\text{s}$ | $3.35\text{s}$ | **HIGH** | Rider passing left-to-right; strong acoustic match ($R = 0.5022$) | [ ] Verified |
| **`pair_012`** | `IMG_0631.MOV` | `13.mp4` | $+1.395\text{s}$ | $4.03\text{s}$ | **HIGH** | Clear road profile; electric wires overhead | [ ] Verified |
| **`pair_013`** | `IMG_0632.MOV` | `14.mp4` | $+6.695\text{s}$ | $6.78\text{s}$ | **HIGH** | Long recording ($15.1\text{s}$ back); Front started $+6.7\text{s}$ later as bike approached | [ ] Verified |
| **`pair_014`** | `IMG_0633.MOV` | `15.mp4` | $-1.820\text{s}$ | $2.66\text{s}$ | **HIGH** | Rider #40 "ARFAN" (NO HELMET); 2 background pedestrians (blue/red shirts) | [ ] Verified |
| **`pair_015`** | `IMG_0634.MOV` | `17.mp4` | $-1.470\text{s}$ | $4.25\text{s}$ | **HIGH** | Smooth trajectory; strong correlation ($R = 0.5079$) | [ ] Verified |
| **`pair_016`** | `IMG_0635.MOV` | `18.mp4` | $-1.675\text{s}$ | $7.58\text{s}$ | **HIGH** | Yellow t-shirt rider with helmet; highest correlation peak ($R = 0.6020$) | [ ] Verified |
| **`pair_017`** | `IMG_0636.MOV` | `19.mp4` | $+0.355\text{s}$ | $3.39\text{s}$ | **HIGH** | Clean background trees; distinct engine deceleration sound | [ ] Verified |
| **`pair_018`** | `IMG_0637.MOV` | `20.mp4` | $+1.405\text{s}$ | $2.62\text{s}$ | **HIGH** | Motorcycle passing speed bump; road dips aligned | [ ] Verified |
| **`pair_019`** | `IMG_0638.MOV` | `21.mp4` | $-2.380\text{s}$ | $3.86\text{s}$ | **MEDIUM** | Ambient traffic sound; check front/back rider shirt color | [ ] **Inspect Medium** |
| **`pair_020`** | `IMG_0639.MOV` | `22.mp4` | $+3.320\text{s}$ | $2.69\text{s}$ | **HIGH** | Fast rider; distinct engine rev; utility pole landmark | [ ] Verified |
| **`pair_021`** | `IMG_0640.MOV` | `23.mp4` | $-1.250\text{s}$ | $4.57\text{s}$ | **HIGH** | Strong acoustic peak ($R = 0.5235$); clear rider posture | [ ] Verified |
| **`pair_022`** | `IMG_0641.MOV` | `24.mp4` | $-1.565\text{s}$ | $3.68\text{s}$ | **MEDIUM** | Slight road bump sound; check rider helmet presence | [ ] **Inspect Medium** |
| **`pair_023`** | `IMG_0642.MOV` | `25.mp4` | $-0.030\text{s}$ | $5.37\text{s}$ | **HIGH** | Near-zero offset ($-30\text{ ms}$); clean $5.37\text{s}$ overlap duration | [ ] Verified |
| **`pair_024`** | `IMG_0643.MOV` | `26.mp4` | $+1.235\text{s}$ | $8.04\text{s}$ | **HIGH** | Longest overlap ($8.04\text{s}$); multiple vehicles passing | [ ] Verified |
| **`pair_025`** | `IMG_0644.MOV` | `27.mp4` | $-0.590\text{s}$ | $3.53\text{s}$ | **HIGH** | Final recording session pair; brick road, pedestrian in white shirt | [ ] Verified |

---

## 4. Specific Inspection for the 3 Medium-Quality Pairs

Three pairs (`pair_008`, `pair_019`, `pair_022`) exhibited cross-correlation values in the range $R \in [0.21, 0.24]$ (due to roadside wind gusts or softer motorcycle engine noise):

1. **`pair_008` Preview:** Open [`pair_008_sync_preview.jpg`](file:///Users/sayemuddin/Desktop/research_helmet/results/phase11_dual_camera/synchronized_preview/pair_008_sync_preview.jpg). Verify the rider's jacket and the pole banner on the left road edge.
2. **`pair_019` Preview:** Open [`pair_019_sync_preview.jpg`](file:///Users/sayemuddin/Desktop/research_helmet/results/phase11_dual_camera/synchronized_preview/pair_019_sync_preview.jpg). Verify the motorcycle tank color and helmet type.
3. **`pair_022` Preview:** Open [`pair_022_sync_preview.jpg`](file:///Users/sayemuddin/Desktop/research_helmet/results/phase11_dual_camera/synchronized_preview/pair_022_sync_preview.jpg). Verify that the rider passing in the back view matches the approaching rider in the front view.

---

## 5. Decision & Sign-Off Procedure

Once you have visually reviewed the preview images:
- If all pairs match: Provide **Approval for Phase 12 (Dual-Camera Evidence Fusion & Cross-Camera Evaluation)**.
- If any individual offset requires micro-adjustment: Note the pair ID and offset adjustment in milliseconds.

---
*Human Verification Guide prepared by Autonomous CV Research Agent.*
