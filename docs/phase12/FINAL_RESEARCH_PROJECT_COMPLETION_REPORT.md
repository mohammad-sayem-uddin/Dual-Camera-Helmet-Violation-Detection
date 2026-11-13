# Final Research Project Completion Report
## Dual-Camera Context-Aware Motorcycle Helmet Violation Detection with Confidence-Aware Temporal Verification

**Date:** 2026-11-13  
**Branch:** `main` (local commits ahead of `origin/main` at `733504e`; not pushed per project instructions)  
**Pre-Report Commit:** `50aa77f` (this report + index entry committed on top of `50aa77f`)  
**Canonical Active Reference:** [`dual_camera_gt_v1.0_ai_assisted_audited`](../../data/dual_camera_test_v1/releases/dual_camera_gt_v1.0_ai_assisted_audited/README.md) (`AI_ASSISTED_AUDITED`, `human_verified = false`)  
**Preserved Future Reference:** [`dual_camera_gt_v1.1_blind_human`](../../data/dual_camera_test_v1/releases/dual_camera_gt_v1.1_blind_human/README.md) (`PENDING_HUMAN_REVIEW`, `executed_human_units = 0 / 92`)

---

## A. Executive Completion Summary & Canonical Reference Policy

This project has completed its full engineering hardening, statistical synthesis, CPU-safe end-to-end reproduction automation, independent final audit, and publication manuscript preparation while honoring all Principal Investigator scope and data-integrity directives:

1. **Accepted Evaluation Reference (`v1.0`):** Per project-owner directive to conclude without completing additional manual review or recruiting external annotators, `dual_camera_gt_v1.0_ai_assisted_audited` (`70` global physical passes across `25` synchronized video pairs and `15,241` decoded frames; `202` identity links; `86` audit correction operations) is accepted as the active benchmark reference, with explicit disclosure of its `AI_ASSISTED_AUDITED` (`human_verified = false`) provenance.
2. **Preserved Unexecuted Blind Package (`v1.1`):** `dual_camera_gt_v1.1_blind_human` (`92` stratified items across `25` pairs; `0 / 1,564` blind response cells populated) and its isolated [`src/review_dashboard/`](../../src/review_dashboard/server.py) web application are preserved untouched with **zero** fabricated or backfilled human labels.
3. **End-to-End Engineering & Verification Closure:** Six targeted software/reproducibility defects (`D1–D6`) were resolved and covered by regression integration tests ([`tests/test_final_system_integration.py`](../../tests/test_final_system_integration.py)); a hardened single-command CPU-safe verification runner ([`scripts/reproducibility/run_full_research_reproduction.sh`](../../scripts/reproducibility/run_full_research_reproduction.sh)) passed all `5/5` mandatory JSON/evaluation gates, all `3` cryptographic SHA-256 manifests (`39/39`, `43/43`, `8/8`), `yolov8n.pt` + `best.pt` weight hashes, frontend JavaScript syntax check (`node v26.9.0`), and all `238` tests in `pytest -q tests/` with **exact** metric parity against committed baselines.
4. **Independent Skeptical Audit Closure:** An independent `research` subagent audit (`3515d430-63cb-4f29-a52d-5b3ef0c99e83`) found **`CRITICAL: 0`**, confirmed all code fixes (`D1–D6`), test suites (`238/238`), SHA-256 checksums, statistical CIs, McNemar tests, and primary tables, and surfaced `5` moderate + `5` minor documentation/CLI discrepancies—all `10/10` of which were resolved and re-verified (`50aa77f`).

---

## B. Git Commit Trajectory (`origin/main` `733504e` $\to$ Completion)

Across the post-`733504e` trajectory on local `main` (without `git push` or history rewriting):

| Stage / Commit | Summary |
|---|---|
| `0ec7c23` | `TASK-09`: Initial `GAP-C` motorcycle-anchored rider tracking, role consensus, and orphan lifecycle implementation |
| `5caabad` – `8cc3c96` | `TASK-09A/09B`: Remediation of conflict-unlock role mismatches, orphan conservation, and independent validation (`Gate A: PASS WITH LIMITATIONS`) |
| `963d238` – `00bd15f` | `TASK-10`: Bounded real-video `Cue C` validation (`pair_005, 008, 009, 019` at `720p` and `4K`) + Gate B validation report |
| `b032707` – `7763cd3` | `TASK-11`: Confidence-aware Beta log-odds dual-camera fusion (`DualCameraFusionEngine`), contradiction gate (`K >= 0.35`), 10-arm ablation + Gate C validation |
| `e38bb17` – `ac761b2` | `TASK-12`: Fixed zero-duration point-interval `tIoU`, blind `v1.1` `92`-item workqueue package, frame-count reconciliation (`15,241`), + Gate D validation |
| `e1d817c` – `e68b742` | Role-isolated local blind GT `v1.1` review & adjudication dashboard + VFR/HEVC frame-exact extraction hardening across two pre-review audits |
| `d64ccbd` | Final defect hardening (`D1–D6`) + [`tests/test_final_system_integration.py`](../../tests/test_final_system_integration.py) |
| `8396fa4` | Hardened unified CPU-safe reproduction script [`run_full_research_reproduction.sh`](../../scripts/reproducibility/run_full_research_reproduction.sh) |
| `5ca1b7c` | Unified final documentation suite (`FINAL_SYSTEM_ARCHITECTURE`, `FINAL_EXPERIMENTAL_METHODOLOGY`, `FINAL_EXPERIMENTAL_RESULTS`, `FINAL_RESEARCH_LIMITATIONS`, `FINAL_PROJECT_COMPLETION_AUDIT`, `ANNOTATION_RELEASE_ACCEPTANCE_AND_PROVENANCE`, `README`, `REPRODUCIBILITY_GUIDE`, `RESEARCH_MANUSCRIPT_DRAFT`) |
| `50aa77f` | Resolution of all `10` independent audit discrepancies (`MOD 1–5`, `MIN 1–5`) and conversion of workstation URIs to portable repo-relative links |
| *Final Commit* | Addition of `docs/phase12/FINAL_RESEARCH_PROJECT_COMPLETION_REPORT.md` and final index link in `README.md` |

Pre-existing untracked file `docs/phase12/GEMINI_FLASH_TASK09_VALIDATION_REPORT.md` (`48,478` B) was left untouched and untracked throughout. Zero tracked files under `results/` or `data/` were altered from their frozen state.

---

## C. Engineering & Defect Remediation (`D1–D6`)

Six concrete code and portability gaps identified during the final codebase audit were fixed in commit `d64ccbd` and verified both individually and end-to-end:

1. **D1 — Cross-Kind Duration-Asymmetry Guard Calibration ([`src/association/orphan_association.py`](../../src/association/orphan_association.py#L122)):**
   - `_score_pair` emitted `"appearance_score"`, whereas `associate_orphans` previously inspected the dead key `"appearance_hsv_score" not in brk`. Naive replacement with `"appearance_score" not in brk` allowed weak uncalibrated opposing-view HSV overlap (`0.558`–`0.674` on `pair_005/009/019`, below Task 10's negative mean `0.6641`) to bypass the extreme cross-kind duration-asymmetry guard (`obs_min < 15` and `obs_min / obs_max < 0.20`), breaking `TASK-10`'s `gt_links_preserved_c2_and_c3` and causing a false link on `pair_005`.
   - **Final Fix:** Reads `app_val = brk.get("appearance_score", brk.get("appearance_hsv_score"))` and blocks extreme cross-kind duration asymmetry whenever `app_val is None` (`_WITHOUT_APPEARANCE`) or `float(app_val) < 0.85` (`_WEAK_APPEARANCE`), while permitting genuinely high positive appearance (`>= 0.85`). Verified across all three regimes in `test_d1_orphan_asymmetry_guard_recognizes_appearance_score_key` and restored `11/11` `True` in `TASK-10` real-video replay.
2. **D2 — `SharedDetectorRegistry` Cache Keying ([`src/detection/camera_detector.py#L149-L163`](../../src/detection/camera_detector.py#L149-L163)):**
   - Keyed cached vehicle and helmet adapters by `f"{device or 'auto'}:{float(conf_threshold):.4f}"` instead of `device` alone so distinct confidence thresholds never collide on stale adapter instances.
3. **D3 — `ema_alpha` Forwarding ([`src/temporal/temporal_engine.py#L31-L39`](../../src/temporal/temporal_engine.py#L31-L39), [`src/temporal/rider_tracker.py#L182-L198`](../../src/temporal/rider_tracker.py#L182-L198)):**
   - Forwarded configured `ema_alpha` (`0.35` default) from `TemporalVerificationEngine.__init__` through `RiderTracker.__init__` into `_create_track`.
4. **D4 — Pipeline Fusion Serialization, CLI, Slot Role & Summary Persistence ([`src/pipeline/dual_camera_pipeline.py`](../../src/pipeline/dual_camera_pipeline.py#L1096-L1103)):**
   - Restored `fusion_mode` and `fusion_config` serialization/deserialization (`_serialize_config`, `_config_from_dict`), exposed `--fusion-mode {legacy,dual_fused,front_only,back_only}` on `main()`, preserved `"consensus_slot_role"` inside `_track_records`, and emitted optional `{pair_id}_fusion_summary.json` exclusively in non-legacy fusion modes (`fusion_mode != "legacy"`), preserving full backward parity for legacy runs.
5. **D5 — Strict `compute_tiou` Mode Validation & Helper Consolidation ([`scripts/reproducibility/evaluate.py#L33-L44`](../../scripts/reproducibility/evaluate.py#L33-L44), [`evaluate_task11_fusion.py#L68-L75`](../../scripts/reproducibility/evaluate_task11_fusion.py#L68-L75)):**
   - Added explicit `ValueError` validation on `mode in ("fixed", "legacy")` in `evaluate.compute_tiou`, unified `tol=1e-9` degenerate zero-duration interval handling, and delegated `evaluate_task11_fusion.py`'s `_tiou_legacy` / `_tiou_fixed` to `evaluate.compute_tiou`.
6. **D6 — Dead Code, Fallback Config Sync & Machine Portability ([`src/association/tracklet_builder.py`](../../src/association/tracklet_builder.py), [`src/tracking/camera_tracker.py`](../../src/tracking/camera_tracker.py), 3 scripts):**
   - Removed unused `_is_bbox_near_border` in `tracklet_builder.py`, synchronized `camera_tracker.py` fallback defaults (`8.0`, `4`) with [`configs/tracking_config.yaml`](../../configs/tracking_config.yaml), and replaced hardcoded `/Users/sayemuddin/Desktop/research_helmet` paths in `audit_both_datasets.py`, `build_smoke_test_notebook.py`, and `materialize_combined_dataset.py` with `Path(__file__).resolve().parents[1]`.

---

## D. Annotation Release Acceptance & Cryptographic Provenance

Detailed in [`ANNOTATION_RELEASE_ACCEPTANCE_AND_PROVENANCE.md`](ANNOTATION_RELEASE_ACCEPTANCE_AND_PROVENANCE.md):

| Release / Artifact | Status | Cryptographic / Record Audit |
|---|---|---|
| [`dual_camera_gt_v1.0_ai_assisted_audited`](../../data/dual_camera_test_v1/releases/dual_camera_gt_v1.0_ai_assisted_audited/README.md) | **Accepted Reference** (`AI_ASSISTED_AUDITED`, `human_verified = false`) | `39/39` base annotation SHA-256 + `43/43` release SHA-256 `OK`; `70` events (`40` violations [`27` driver, `10` both, `3` pillion], `6` compliant, `24` `INSUFFICIENT_VISUAL_EVIDENCE`; `29` front-only, `27` back-only, `10` linked-both, `4` ambiguous), `202` identity links, `86` correction ops (`63` pass2 + `6` pass3 + `17` stage_c), `14,249` `rider_frame_validity.csv` rows (`6,727` valid), `178` uncertainty rows, `0/70` plates assessed |
| [`dual_camera_gt_v1.1_blind_human`](../../data/dual_camera_test_v1/releases/dual_camera_gt_v1.1_blind_human/README.md) | **Preserved Template** (`PENDING_HUMAN_REVIEW`, `0/92`) | `release_metadata.json` (`3e270c9ae8901a...`): `human_verified = false`, `executed_human_units = 0`; `blind_annotation_workqueue.csv` (`27c18280b3cf...`): `92` items (`70` physical + `22` GAP-C split), `0 / 1,564` blind response cells filled, `92/92 PENDING_HUMAN_REVIEW`; `review_workspace/` contains only `README.md` (`142c6f06e167...`) |
| Stage K Frozen Bundle (`results/phase12_integrated/`) | **Frozen Reference** | `8/8` SHA-256 checksums `OK` in [`results/phase12_integrated/checksums.sha256`](../../results/phase12_integrated/checksums.sha256) |
| Model Checkpoints | **Frozen Reference** | `yolov8n.pt`: `f59b3d83...cf83b36`; `results/phase9_multiseed/exp_combined/seed_42/weights/best.pt` and `results/exp_combined/weights/best.pt`: `0929a882...40b0e5` |

---

## E. Integrated System Architecture Summary

Described in full with equations and Mermaid flowcharts in [`FINAL_SYSTEM_ARCHITECTURE.md`](FINAL_SYSTEM_ARCHITECTURE.md):

1. **Sequential Container PTS Sync (`Phase 12C`, [`pts_index.py`](../../src/synchronization/pts_index.py)):** Avoids `CAP_PROP_POS_FRAMES` random seeks (eliminating up to `2,870.00 ms` VFR error in `pair_013_back.mp4`); aligns `CAP_PROP_POS_MSEC` to `ffprobe` rational PTS within `0.000000 ms` across all `50` videos (`15,241` total packets/frames: `6,557` Front + `8,684` Back) and separates $\Delta t_{\text{rec}} \in [-2.380, +6.695]\text{ s}$, $\delta_{\text{pts}}$, $\epsilon_{\text{sync}} \in [3.0, 38.0]\text{ ms}$, and rolling-shutter exposure phase.
2. **Per-Camera Dual-Model Detection (`Phase 12D`, `D2`):** Runs COCO `yolov8n.pt` (`person`, `motorcycle`, thresholds `0.45`/`0.18`) and multi-domain `EXP-COMBINED` `seed_42` `best.pt` (`no_helmet`/`helmet`, threshold `0.25`) inside a device + threshold isolated registry.
3. **Hierarchical Motorcycle-Anchored Rider Tracking (`GAP-C`, `TASK-09/09A`):** Couples heads to motorcycles (`MotoRiderCoupler`, max `3` riders), propagates heads in dimensionless motorcycle-normalized coordinates $(u, v, \tilde{w}, \tilde{h})$ with `rel_damping_factor=0.20` (dataclass field `vel_damping`), unlocks multi-rider conflict stalls (`sibling_overlap_max_iou=0.35`, `conflict_distinct_min_rel_dist=0.58`) over finite coasting (`15` confirmed / `3` tentative frames), maintains longitudinal multi-frame `driver`/`pillion_1` consensus, and gates confirmed-only orphans (`min_hits=2`, `max_lost=4`).
4. **Hysteresis Temporal Verification & Genuine Evidence Accumulation (`Phase 10`, `TASK-04`, `D3`):** Combines an online 6-state hysteresis FSM ($\alpha = 0.35$, escalation $\ge 0.70$ for `5` frames, de-escalation $< 0.40$) with `HelmetEvidenceAccumulator` (`conf_weighted`) over genuinely coupled detections so coasting windows never zero out terminal track decisions.
5. **Protected Keyframes & Multi-Cue Association (`TASK-06/06A`, `Phase 12E`, `D1`):** Retains `<= 24` frames/camera while protecting sole-observation anchors of co-occurring tracks, maps `3840x2160` boxes into `720p` crops via [`crop_native_box`](../../src/association/coordinate_space.py), and matches across cameras with missing-cue renormalization (`min_active_cues_required=2`) across `Cue A` (transit `0.45`), `Cue B` (cardinality `0.25`), optional `Cue C` (HSV+V `0.15`), `Cue E` (convoy `0.10`), `Cue D` disabled, Hungarian solver (`>= 0.62`, margin `>= 0.10`), and strong-appearance asymmetry guarding (`>= 0.85`).
6. **Sync-Discounted Log-Odds Fusion & Contradiction Gating (`TASK-11`, `D4`):** Forms Beta-smoothed posteriors $P_c$ and logits $L_c$ weighted by maturity, head area scale, border, flip stability, viewpoint prior ($\gamma_F=1.10, \gamma_B=0.95$), and confirmation multiplier (`1.08`), discounts secondary views by $c_{\text{assoc}} = s_{\text{assoc}} \cdot \max(0.80, \exp(-0.003 \cdot \max(0, \epsilon_{\text{sync}} + \bar{\delta}_{\text{pts}} - 16.67)))$, adds directional corroboration ($\beta = 0.20$), and abstains (`EVIDENCE_CONFLICT_ABSTAIN`) whenever $K_{\text{conflict}} = 2\min(|P_F - 0.5|, |P_B - 0.5|) \ge 0.35$.

---

## F. Single-Camera Dataset Integrity & Multi-Seed Detector Training (`Phases 7–9`)

- **Sequence-Isolated Old CCTV (`5,547` images / `169` sequences):** `3,866` train, `852` val, `829` held-out test (`1,170` boxes); `0.0%` sequence leakage.
- **Refined Mobile Street Expansion (`1,841` images):** `1,285` train, `276` val, `280` test (`552` boxes); `392` torso-bleeding boxes refined to cranial bounds across `318` images (max aspect ratio `4.20 -> 1.47` across `3,733` boxes; `0` class flips).
- **Combined Leak-Free Union (`7,388` images):** `5,151` train, `1,128` val, `1,109` held-out test (`1,722` boxes).
- **Phase 9 Three-Seed Reproducibility (`42, 123, 2026`, `mean_std_summary.csv`):**
  - `EXP-NEW` vs. `EXP-COMBINED` on **Old CCTV (`829` images)**: `mAP@0.50` improves from `0.1164 ± 0.0419` to **`0.7595 ± 0.0090`** (`+64.31 pp`, Cohen's $d = +19.55$), F1 from `0.2009 ± 0.0433` to **`0.7585 ± 0.0084`** ($d = +15.89$).
  - On **New Mobile (`280` images)**: `mAP@0.50` remains virtually unchanged (`0.9609 ± 0.0029` vs. `0.9601 ± 0.0015`, $-0.08\text{ pp}$).
  - On **Combined (`1,109` images)**: `mAP@0.50` increases from `0.4209 ± 0.0282` to **`0.8477 ± 0.0047`** ($+42.68\text{ pp}$, $d = +18.16$) and F1 from `0.5011 ± 0.0168` to **`0.8144 ± 0.0040`** ($+31.32\text{ pp}$, $d = +23.55$).

---

## G. Temporal Verification & Track-Level Redesign (`Phase 10`, `TASK-04`)

- **Phase 10 Synthetic Benchmark (`21` runs, `benchmark_comparison_table.csv`):** All `3` aggregators achieve `Precision = 1.00`, `Recall = 1.00`, `0` false alerts, `Stability = 1.000`, `5.0`-frame (`0.167 s` at 30 fps) violation delay on violation/occlusion scenarios, and abstention rates `0.097`–`0.115`.
- **TASK-04 Genuine-Observation Track Redesign (`helmet_track_level_redesign.json`, `n=209` decided tracks):** Eliminates all `73` legacy `ema_final` `WINDOW_FLUSH_ZERO_MASS` abstentions (`0.6411` [`134/209`] $\to$ **`0.9904`** [`207/209`, `0` abstentions] under `conf_weighted`), achieves **`1.0000`** (`85/85`) on $K=1$ tracks (vs. `0.6471` legacy), and causes `0 / 85` regressions across frozen Stage J events (`events_changed = 0`).

---

## H. Dual-Camera Real-Video Evaluation (`Stage K`, `GAP-B`, `GAP-C`, `TASK-10..12`)

1. **Stage K & `TASK-12` Fixed `compute_tiou` ([`stagek_results.json`](../../results/phase12_integrated/stagek_results.json), [`task12_tiou_comparison.json`](../../results/phase12_task12/task12_tiou_comparison.json)):**
   - Head detection operating point: Front `F1=0.9253` (`AP50=0.8932`), Back `F1=0.5145` (`AP50=0.3467`, `2,138` FP), Pooled `F1=0.8161` (`[0.7454, 0.8762]` bootstrap CI).
   - Circular Stage J consistency metrics: spatially valid coupled obs `F1=1.0000` (`6,727`), head-basis `IDF1=0.9987` (`0` switches).
   - Association: Motorcycle `F1=0.9091` (`TP=10, FP=2, FN=0`, `38.82%` abstain), Orphan `F1=1.0000` (`n=1` link, `63.33%` abstain).
   - Violation Events (`N=70`): Legacy `compute_tiou` `F1=0.8258` (`64 TP / 21 FP / 6 FN`) $\to$ Fixed `TASK-12` point-interval `compute_tiou` **`F1=0.8516`** (`66 TP / 19 FP / 4 FN`, Wilson P `[0.677, 0.852]`, R `[0.862, 0.978]`; `DRIVER_NO_HELMET` `0.8519 -> 0.9259`; `0.0 s` timing MAE).
2. **GAP-B Keyframes, Coordinate Scaling & Evidence Gate (`TASK-06/06A`, [`gapb_evaluation_results.json`](../../results/phase12_gapb_redesign/gapb_evaluation_results.json)):**
   - Stage J stored descriptors for only `7 / 100` (`7%`) motorcycles. Within the `<= 24` cap, protected-anchor retention reached `98 / 100` on pre-GAP-C frozen tracklets (`2` sim desyncs solved by `TASK-06A`) and **`100 / 100`** (`0` stale refs, `9 / 9` PNG crops recovered under `4K -> 720p` scaling) in live post-GAP-C replay, eliminated `93` legacy `1080p` border FPs (`86 -> 35` entry, `75 -> 33` exit), and recovered `74/74` frozen (`81/83` live post-GAP-C) `UNCERTAIN` rider subtracks and `29/29` frozen (`14/14` live) orphan tracks with `0` active-cue gate violations.
3. **`TASK-10` Bounded Real-Video `Cue C` (`pair_005, 008, 009, 019` at `720p` & `4K`, [`task10_cue_c_results.json`](../../results/phase12_task10/task10_cue_c_results.json)):**
   - Executed in `17.279 s` (`1,806.03 MB` RSS): raised descriptor availability from `8.0%` (`6/75`) to **`100.0%` (`55/55`)**, confirmed **`0.9971`** `720p vs 4K` self-similarity (`|diff|` mean `0.0083`, max `0.0278`), and established negative opposing-view histogram overlap (`0.7223` positive `n=5` vs. `0.6641` negative `n=20`), nudging borderline `pair_005` (`0.6192`) to `0.6203` (`+0.0003` false `LINKED` at `4K`).
4. **`GAP-C` 5-Arm Tracking Ablation (`A0` $\to$ `A4`, [`gapc_ablation_summary.csv`](../../results/phase12_gapc/gapc_ablation_summary.csv)):**
   - Back coupled head retention: **`28.69%` $\to$ `91.66%`** (`+62.97 pp`, `860 -> 2,748 / 2,998`); Front: `84.81% -> 95.87%`; Combined: `66.61% -> 94.50%`.
   - Abstained updates cut `92.2%` (`2,943 -> 230`); out-of-bounds boxes `660 -> 0`; `A2` conflict role mismatches `47` (`13` multi-driver) suppressed back to **`0`** (`0` multi-driver) via `A3`; `1`-hit / low-conf `<0.45` orphan noise reduced `56 / 34 -> 0 / 0`; `18/18` acceptance criteria `true`.
5. **`TASK-11` 10-Arm Dual-Camera Fusion (`70` passes / `40` violations / `46` decided, [`task11_ablation_summary.csv`](../../results/phase12_task11/task11_ablation_summary.csv)):**
   - Best unified fusion arm **`A4_DUAL_FUSED_SPLIT`** achieved **`97.14%` (`68/70`, Wilson `[0.902, 0.992]`)** physical pass coverage, **`82.50%` (`33/40`, `[0.681, 0.913]`)** binary violation recall, **`65.22%` (`30/46`, `[0.508, 0.773]`)** decided accuracy, and **`0.6582`** strict fixed-tIoU F1 (`1` conflict abstain on `pair_015`, `8` quarantine splits).
   - **Exact two-sided McNemar significance:** `A4_DUAL_FUSED_SPLIT` beats `A4_FRONT_ONLY` on physical coverage (`+27/-1`, $p = 2.16 \times 10^{-7}$), violation recall (`+10/-1`, $p = 0.0117$), and strict TP (`+23/-0`, $p = 2.38 \times 10^{-7}$); beats `A4_BACK_ONLY` on physical coverage (`+29/-1`, $p = 5.77 \times 10^{-8}$), violation recall (`+15/-0`, $p = 6.10 \times 10^{-5}$), decided accuracy (`+20/-2`, $p = 1.21 \times 10^{-4}$), and strict TP (`+33/-0`, $p = 2.33 \times 10^{-10}$).
   - Resolves `3/3` Stage J $K=1$ vetos (`pair_012, 018, 024`), quarantines opposing false links (`pair_015`, $K=0.35$), links `pair_009` orphan (`P=0.7307`), and rescues `Back-Only` violation recall (`35.0% -> 45.0%`). Strict `A4` vs `A0` schema delta (`50 vs 63`, $p = 0.00098$) reflects `v1.0` labeling relative to pre-GAP-C `A0` subtracks (`GAP-C` recovers previously dropped pillions and updates slot consensus).

---

## I. Independent Final Audit & Resolution of Discrepancies

An independent read-only `research` subagent audit (`3515d430-63cb-4f29-a52d-5b3ef0c99e83`) at commit `5ca1b7c` reported **`CRITICAL: 0`**, verified all `D1–D6` implementations, `238/238` tests, all SHA-256 manifests, `v1.1` `0/92` integrity, all reproduction gates, Wilson CIs, and McNemar p-values, and flagged `10` documentation/CLI issues—every one of which was corrected in commit `50aa77f`:

- **MOD 1 (`TASK-10` pair scope):** Corrected `pair_001, 013, 014, 015` mislistings to exact `pair_005, pair_008, pair_009, pair_019` across both `720p` and `4K` in `FINAL_EXPERIMENTAL_METHODOLOGY.md`, `FINAL_EXPERIMENTAL_RESULTS.md`, and `FINAL_PROJECT_COMPLETION_AUDIT.md`.
- **MOD 2 (`expansion-v2` `392` boxes):** Corrected `"28 inversions"` / `"removing 392 boxes"` wording to accurately reflect refining `392` torso-bleeding boxes across `318` images to cranial boundaries (`3,733` boxes preserved, `0` class flips per [`BBOX_PRECISION_CORRECTION_FINAL_REPORT.md`](../../BBOX_PRECISION_CORRECTION_FINAL_REPORT.md)) in `RESEARCH_MANUSCRIPT_DRAFT.md`, `FINAL_EXPERIMENTAL_METHODOLOGY.md`, and `PAPER_OUTLINE.md`.
- **MOD 3 (CLI flags):** Updated `--pair-id` / `--output-dir` examples in `README.md` and `docs/REPRODUCIBILITY_GUIDE.md` to actual `dual_camera_pipeline.py` flags `--pairs` / `--output-root` (`--help` verified exit `0`).
- **MOD 4 (`15,241` frames):** Removed legacy `15,637` packet figure from `FINAL_EXPERIMENTAL_METHODOLOGY.md`, aligning with [`PHASE12_FRAME_COUNT_RECONCILIATION.md`](PHASE12_FRAME_COUNT_RECONCILIATION.md) (`15,241`).
- **MOD 5 (`GAP-B` frozen vs. live scope):** Explicitly separated frozen pre-GAP-C Stage J counts (`98/100`, `74/74` recovered, `86` riders, `130` orphans) from live post-GAP-C `evaluate_gapb_redesign.py` replay (`100/100`, `81/83` recovered with `2` remaining `UNCERTAIN` on `95` riders / `68` orphans).
- **MIN 6–10:** Corrected `pair_004` `occluded_unknown` observation count (`6` observations vs. `5` event-level slots), $\epsilon_{\text{sync}}$ minimum (`3.0 ms`), `rel_damping_factor` vs. `vel_damping` distinction, `57.14%` back-only coverage citation in `README.md`, untracked report link removal in `FINAL_PROJECT_COMPLETION_AUDIT.md`, `e68b742` audit-basis note in `ANNOTATION_RELEASE_ACCEPTANCE_AND_PROVENANCE.md`, and converted `155` workstation `file:///Users/...` URIs into portable repo-relative links (`180` links checked, `broken = 0`).

---

## J. Reproducibility Verification Matrix & Cryptographic Summary

Full CPU-safe execution via [`scripts/reproducibility/run_full_research_reproduction.sh`](../../scripts/reproducibility/run_full_research_reproduction.sh) (`/tmp/research_repro_final`):

| Check | Outcome | Verified Measurement |
|---|---|---|
| Environment | `PASS` | Python `3.11.15` (`/opt/homebrew/opt/python@3.11/bin/python3.11`), PyTorch `2.14.1` (`mps_available=True`), Ultralytics `8.4.201`, OpenCV `5.0.0`, SciPy `1.17.1`, `node v26.9.0` |
| SHA-256 Manifests & Weights | `PASS` | Annotations `39/39`; `v1.0` Release `43/43`; Stage K `8/8`; `yolov8n.pt` + `best.pt` matched |
| `v1.1` Untouched Check | `PASS` | `PENDING_HUMAN_REVIEW`, `executed_human_units=0`, `0/1,564` cells |
| `[3/11]` Stage K Legacy + `[4/11]` Fixed | `PASS` | Legacy F1 `0.8258` -> Fixed F1 `0.8516`; byte-exact CSV parity (`events_detail.csv`, `sync_by_pair.csv`) |
| `[5/11]` `GAP-B` Live Replay | `PASS` | `gapb_item_level_100pct=true` (`1.0` coverage, `1.0` F1 on `17` retained items), `100/100` keyframe coverage |
| `[6/11]` `GAP-C` Ablation | `PASS` | `gapc_all_acceptance_criteria_passed=true` (`18/18` checks; zero metric delta vs. committed baseline) |
| `[7/11]` `TASK-10` Real-Video `Cue C` | `PASS` | `task10_all_checks_passed=true` (`11/11` checks; zero metric delta vs. committed baseline) |
| `[8/11]` `TASK-11` 10-Arm Fusion | `PASS` | `task11_all_checks_passed=true` (`10/10` checks; zero metric delta vs. committed baseline) |
| `[9/11]` `TASK-12` `tIoU` & `v1.1` Audit | `PASS` | `task12_all_checks_passed=true` (`6/6` checks; exact match) |
| `event_regression_task04.py` | `PASS` | `events compared=85 changed=0`; zero diff in `results/` |
| `[10/11]` Dashboard JS Syntax | `PASS` | `node --check src/review_dashboard/static/app.js` (`v26.9.0`) |
| `[11/11]` Full `pytest` Suite | `PASS` | `238 passed, 5 warnings in 107.82s` (`reproduction_manifest.json` `"status": "PASSED"` on 5/5 gates; subagent re-verified `238 passed` at `5ca1b7c`) |
| Cross-Doc Markdown Link Check | `PASS` | `180` links across `10` docs (`broken = 0`; zero workstation `file://` URIs) |

*(Intentionally excluded heavy compute stages: Phase 7/9 Kaggle T4 GPU detector retraining, Stage J full `50`-video inference regeneration, and `v1.1` manual multi-reviewer annotation execution.)*

---

## K. Consolidated Scientific Limitations & Next Research Steps

As synthesized in [`FINAL_RESEARCH_LIMITATIONS.md`](FINAL_RESEARCH_LIMITATIONS.md):
1. **AI-Assisted Reference Circularity (`v1.0`):** `v1.0` originated from Pass 1 detections/association before `86` edits (`39.3%` Pass-3 frame-layer error rate, mainly `46/66` stale box propagation; `50.6%` non-coupled subtrack rows). Therefore, single-camera/tracking metrics establish internal consistency and controlled ablation deltas, not independent human-ground-truth generalization.
2. **Unexecuted `v1.1` Inter-Rater Consensus (`0/92`):** Because blind multi-reviewer execution was halted per PI priority, `A4` vs `A0` differences on pre-GAP-C `v1.0` schema (`p = 0.00098`) cannot yet be re-scored against independent human consensus.
3. **Unlabeled License Plates (`0/70`):** ALPR/OCR ([`ocr_adapter.py`](../../src/models/ocr_adapter.py)) remains unit-tested and configuration-ready but empirically unmeasured.
4. **Weak Opposing-View HSV `Cue C` & Disabled `Cue D`:** Uncalibrated histograms overlap across opposing views (`0.7223` vs. `0.6641`, `pair_005` 4K `+0.0003` FP), and spatial homography (`Cue D`) remains disabled.
5. **Compute Scope & Small-Sample `v1.0` Substrata:** Only `10/70` `v1.0` passes are `LINKED_BOTH_VIEWS`, `1/1` orphan GT links exist, and local CPU reproduction replays frozen Stage J detections rather than re-running full 50-stream neural inference.

**Highest-Value Future Extensions:** (i) run the `92`-item role-isolated [`dual_camera_gt_v1.1_blind_human`](../../data/dual_camera_test_v1/releases/dual_camera_gt_v1.1_blind_human/README.md) campaign using [`./scripts/start_gt_review_dashboard.sh`](../../scripts/start_gt_review_dashboard.sh); (ii) train calibrated opposing-view appearance embeddings; and (iii) annotate and benchmark rear-plate OCR end-to-end.

---

## L. Index of Final Repository Deliverables

| Deliverable | Path | Purpose |
|---|---|---|
| Project Completion Report | [`docs/phase12/FINAL_RESEARCH_PROJECT_COMPLETION_REPORT.md`](FINAL_RESEARCH_PROJECT_COMPLETION_REPORT.md) | Master engineering, audit, statistical, and reproducibility completion record |
| Release Acceptance & Provenance | [`docs/phase12/ANNOTATION_RELEASE_ACCEPTANCE_AND_PROVENANCE.md`](ANNOTATION_RELEASE_ACCEPTANCE_AND_PROVENANCE.md) | Formal `v1.0` acceptance, `v1.1` preservation, SHA-256 inventory |
| Architecture Documentation | [`docs/phase12/FINAL_SYSTEM_ARCHITECTURE.md`](FINAL_SYSTEM_ARCHITECTURE.md) | Full multi-layer architecture, equations, Mermaid diagrams, config/code bindings |
| Experimental Methodology | [`docs/phase12/FINAL_EXPERIMENTAL_METHODOLOGY.md`](FINAL_EXPERIMENTAL_METHODOLOGY.md) | Datasets, multi-seed setup, sync forensics, `Stage K` & `GAP-B`/`GAP-C`/`TASK-10..12` protocols |
| Experimental Results & Stats | [`docs/phase12/FINAL_EXPERIMENTAL_RESULTS.md`](FINAL_EXPERIMENTAL_RESULTS.md) | Reconciled tables, 95% Wilson/bootstrap CIs, McNemar tests, forensic cases |
| Research Limitations | [`docs/phase12/FINAL_RESEARCH_LIMITATIONS.md`](FINAL_RESEARCH_LIMITATIONS.md) | Methodological caveats, `v1.0` circularity, negative findings, honest non-claims |
| Engineering Audit Register | [`docs/phase12/FINAL_PROJECT_COMPLETION_AUDIT.md`](FINAL_PROJECT_COMPLETION_AUDIT.md) | Defect `D1–D6` audit register and remediation record |
| Reproducibility Guide & Runner | [`docs/REPRODUCIBILITY_GUIDE.md`](../REPRODUCIBILITY_GUIDE.md), [`run_full_research_reproduction.sh`](../../scripts/reproducibility/run_full_research_reproduction.sh) | Environment pin, CLI commands, and 11-step automated verification gate |
| Defect Integration Suite | [`tests/test_final_system_integration.py`](../../tests/test_final_system_integration.py) | Dedicated unit/integration tests covering `D1–D6` (`6` tests) |
| Publication Manuscript Draft | [`docs/paper/RESEARCH_MANUSCRIPT_DRAFT.md`](../paper/RESEARCH_MANUSCRIPT_DRAFT.md) | Complete academic paper draft with `18` verified citations and provenance disclosure |
| Repository Entry Point | [`README.md`](../../README.md) | High-level system overview, quick start, verified highlights, and full index |
