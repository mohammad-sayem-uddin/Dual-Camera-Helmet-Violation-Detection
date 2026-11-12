# ARGON TASK-12 ARCHITECTURE & PROTOCOL — Blind Human Ground Truth `v1.1` (`dual_camera_gt_v1.1_blind_human`) & Evaluation Fixes

**Date:** 2026-11-12  
**Architecture Authority:** Gemini Argon  
**Prerequisite Cleared Gates:** `TASK-09B` (`8cc3c96`), `TASK-10` (`00bd15f`), `TASK-11` (`7763cd3`)

---

## 1. Scientific Rationale & Motivation

`dual_camera_gt_v1.0_ai_assisted_audited` (the frozen Stage E ground-truth release) explicitly discloses that its labels (`AI_ASSISTED_AUDITED`, `human_verified: false`) were generated from Pass-1 Stage J `A0` detections followed by `86` AI audit correction ops (`69` Mission-1 + `17` Stage C), with **zero true human-annotated bounding boxes/labels** and **70/70 (`100%`) license plates marked `NOT_ASSESSED`**.

Experiments completed across `TASK-09A` through `TASK-11` exposed three concrete limitations of benchmarking future upgrades solely against `v1.0`:
1. **AI-Circularity Bias Against Upstream Tracker Improvements:** When `GAP-C` (`A4_DUAL_FUSED_SPLIT`) unlocked Back-camera coupled pillion tracking (`rider_abstained_updates` `-92.2%`, Back-only violation binary recall `+10.0 pp`), strict F1 against `v1.0` appeared lower (`0.6582` vs `A0_DUAL_LEGACY` `0.8182`) primarily because `v1.0` had never annotated newly recovered genuine pillion riders across at least **8 pairs** (`pair_003`, `pair_004`, `pair_013`, `pair_018`, `pair_020`, `pair_021`, `pair_022`, `pair_024`, plus `pair_007` split), plus residual unlinked track discrepancies (`pair_002`, `pair_005`, `pair_009`, `pair_010`, `pair_011`, `pair_012`, `pair_014`, `pair_015`, `pair_016`) and `24` `INSUFFICIENT_VISUAL_EVIDENCE` events.
2. **Unassessed Rear License Plates:** All `70/70` GT events in `v1.0` carry `plate_visibility_tier: NOT_ASSESSED` and `gt_plate_text_normalized: null`, preventing evaluation of end-to-end enforcement attribution or cross-camera plate linkage safety (e.g., blocking quarantined ambiguous links like `pair_020`).
3. **Evaluator Point-Interval tIoU Defect (`scripts/reproducibility/evaluate.py`):** Legacy `tiou(a, b)` returned `0.0` whenever `union == 0.0`, penalizing exact 1-frame violation matches (`[t, t]` vs `[t, t]`) in `pair_012/front_moto_0003` (`[1.428344, 1.428344]`) and `pair_015/front_moto_0001` (`[-1.420111, -1.420111]`) as `FP + FN` despite `0.0 s` onset/end error and exact identity/class match.

---

## 2. Core Principles for `dual_camera_gt_v1.1_blind_human`

### 2.1 Non-Destructive Separation & Anti-Fabrication Policy
1. **Immutable `v1.0` Baseline:** `data/dual_camera_test_v1/releases/dual_camera_gt_v1.0_ai_assisted_audited/` and `data/dual_camera_test_v1/annotations/` SHALL NEVER be modified, edited, or overwritten (`43/43` + `39/39` SHA-256 invariants preserved indefinitely).
2. **Isolated `v1.1` Release Tree:** All blind human annotation packaging, schema definitions, workqueues, and future verified outputs reside strictly under `data/dual_camera_test_v1/releases/dual_camera_gt_v1.1_blind_human/`.
3. **Honesty Invariant (`PENDING_HUMAN_REVIEW`):** No LLM, synthetic heuristic, or pipeline prediction may ever be written as a human annotation. Until real human reviewers complete Passes 1–3, `release_metadata.json` and `schema.json` MUST record `"human_verification_status": "PENDING_HUMAN_REVIEW"`, `"human_verified": false`, `"executed_human_units": 0`, and blank response columns in `blind_annotation_workqueue.csv`.

### 2.2 Blinding & Presentation Protocol (Pass 1)
- **Zero AI Leakage:** Annotators `A` and `B` receive synchronized `(pair_id, front_video, back_video, offset_sec, epsilon_sync_ms)` clips rendered directly from raw `.mp4` frames overlaid ONLY with frame index and synchronized `t_ref_sec`. **No YOLO boxes, ByteTrack IDs, GAP-C subtracks, Stage J decisions, or TASK-11 posteriors are displayed.**
- **Dual-View Synchronized Review UI:** Annotators inspect side-by-side Front + Back native streams stepped at synchronized `t_ref_sec` within the bounded misalignment interval `epsilon_sync_ms` (and labeled drift bound `epsilon_sync_ms + |drift_ms|` for `pair_004/005/008/015/016/019`).
- **Independent Physical Pass Segmentation:** Each physical motorcycle transit is delimited on `(t_enter_ref_sec, t_exit_ref_sec)` per visible camera, assigned a clean session-local transit index, and annotated independently by Annotator A and Annotator B before cross-camera linking.

### 2.3 Two-Annotator + Independent Adjudicator Workflow (Pass 1 -> Pass 3)
1. **Pass 1 (`Annotator A` & `Annotator B` Blind Independent Coding):**
   - For every visible physical motorcycle transit and every visible occupant (`driver`, `pillion_1`, `pillion_2`):
     - Role stability verified against camera-mount vertical & motorcycle travel axis (bottom-most/forward rider on bike seat = `driver`, aft/upper = `pillion_*`).
     - Human headgear label per rider across full visible transit: `helmet`, `no_helmet`, or `occluded_unknown` (never forced from single blurry frame).
     - Violation temporal window `[w0_ref_sec, w1_ref_sec]` recorded where at least one `no_helmet` frame is visually confirmed; 1-frame confirmations explicitly recorded as `[t, t]`.
     - Cross-camera identity linkage (`LINKED_BOTH_VIEWS`, `FRONT_VIEW_ONLY`, `BACK_VIEW_ONLY`, `AMBIGUOUS_HUMAN_UNCERTAIN`).
     - Back-camera rear plate assessment (`plate_visibility_tier` + normalized text when `PLATE_CLEAR`/`PLATE_PARTIAL`, `null` otherwise):
       - `PLATE_CLEAR` (all characters legibly readable by both annotators)
       - `PLATE_PARTIAL` (subset of characters readable, others motion-blurred/occluded)
       - `PLATE_UNREADABLE_MOTION_BLUR` (plate region present in Back view, characters illegible)
       - `PLATE_NOT_VISIBLE_FRONT_ONLY` (no Back camera transit available)
2. **Pass 2 (Inter-Annotator Agreement & Cohen's $\kappa$ Audit):**
   - Compute raw agreement and unweighted Cohen's kappa ($\kappa = \frac{p_o - p_e}{1 - p_e}$) across all completed workqueue units for:
     - Cross-camera link status (`LINKED_BOTH_VIEWS` / `FRONT_ONLY` / `BACK_ONLY` / `AMBIGUOUS`)
     - Event-level binary violation (`violation` vs `non-violation` vs `insufficient_evidence`)
     - Event 5-class label (`DRIVER_NO_HELMET`, `PILLION_NO_HELMET`, `BOTH_NO_HELMET`, `NO_VIOLATION`, `INSUFFICIENT_VISUAL_EVIDENCE`)
     - Per-rider role (`driver` vs `pillion_*`) & helmet label (`helmet` / `no_helmet` / `occluded_unknown`)
     - Plate visibility tier & normalized text exact match (when `PLATE_CLEAR`).
   - Acceptance target before freezing `v1.1` as human-verified: **$\kappa \ge 0.85$** on event binary violation and rider helmet state, **$\kappa \ge 0.90$** on cross-camera link status.
3. **Pass 3 (Blind Adjudication):**
   - Every item where `Annotator A != Annotator B` — plus a mandatory stratified `100%` review of High-Priority Workqueue Strata (`GAP-C` recovered pillion pairs, `24` `v1.0` `INSUFFICIENT_VISUAL_EVIDENCE` events, Stage J `AMBIGUOUS_ABSTAIN` links like `pair_020`, `pair_015` false-link candidate, `pair_009` orphans, and `70/70` Back-view plates) — is adjudicated by a third independent `Adjudicator` viewing raw native synchronized video without seeing which model (`A0` vs `A4`) disagreed.

---

## 3. Workqueue Stratification (`blind_annotation_workqueue.csv`)

Every workqueue row represents one anonymized physical pass candidate indexed by `(workqueue_id, pair_id, sync_quality, priority_stratum, front_clip_t_ref_hint, back_clip_t_ref_hint)` with zero AI verdict leakage into annotator response columns:
- **`STRATUM_1_GAPC_DIVERGENCE`:** Pairs/passes where GAP-C (`A4_DUAL_FUSED_SPLIT`) & `A0` / `GT v1.0` diverged on rider count, pillion recovery, or event classification (`pair_003`, `pair_004`, `pair_013`, `pair_018`, `pair_020`, `pair_021`, `pair_022`, `pair_024`, `pair_007`, plus residual delta pairs `pair_002`, `pair_005`, `pair_009`, `pair_010`, `pair_011`, `pair_012`, `pair_014`, `pair_015`, `pair_016`).
- **`STRATUM_2_INSUFFICIENT_AND_AMBIGUOUS`:** `v1.0` passes with `INSUFFICIENT_VISUAL_EVIDENCE` (`24` events), `AMBIGUOUS_HUMAN_UNCERTAIN` (`4` events including `pair_020`), and unlinked orphan/conflict inspections (`pair_009`, `pair_015`).
- **`STRATUM_3_BASELINE_VERIFICATION_AND_PLATES`:** Remaining `v1.0` decided physical passes to confirm 100% corpus coverage (`70` baseline physical passes + `22` GAP-C recovery / split / excluded candidate transits = `92` physical workqueue items across all `25` pairs: `46` Stratum 1, `33` Stratum 2, `13` Stratum 3) AND upgrade `70/70` `NOT_ASSESSED` license plates.

---

## 4. Point-Interval tIoU Correction in `scripts/reproducibility/evaluate.py`

In [`scripts/reproducibility/evaluate.py`](file:///Users/sayemuddin/Desktop/research_helmet/scripts/reproducibility/evaluate.py), legacy `tiou(a, b)` checked `return inter / union if union > 0 else 0.0`. For 1-frame violation intervals `[t, t]`, matching `a = [t, t]` and `b = [t, t]` produces `inter = 0.0` and `union = 0.0`, erroneously scoring exact matches (`d_onset = 0.0`, `d_end = 0.0`) as `0.0 < 0.5`.

**Specification (`compute_tiou(a, b, *, mode='fixed', tol=1e-9)`):**
- If any bound is `None`: return `None`.
- If `mode == 'legacy'`: preserve historical `inter / union if union > 0 else 0.0` for exact bit-level reproduction audits.
- If `mode == 'fixed'` (new default): when `union <= tol`, return `1.0` if `abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol` else `0.0`; otherwise return `inter / union`.

**Verified Non-Destructive Reporting:** `results/phase12_integrated/` remains 100% immutable (`8/8` SHA-256 preserved); side-by-side legacy vs fixed outputs are stored cleanly under `results/phase12_task12/`.
