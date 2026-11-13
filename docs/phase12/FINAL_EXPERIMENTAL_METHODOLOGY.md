# Final Experimental Methodology

**Project:** An Integrated Dual-Camera Framework for Context-Aware Motorcycle Helmet Violation Detection with Confidence-Aware Temporal Verification  
**Repository:** `Dual-Camera-Helmet-Violation-Detection` (`main`)

---

## 1. Datasets, Sequence Isolation & Split Integrity

### 1.1 Single-Camera Cross-Domain Training & Evaluation Corpus (Phases 1–9)
- **Historical Fixed-Angle CCTV (`old-dataset-v1-fixed` / [`configs/dataset_group_split_fixed.yaml`](../../configs/dataset_group_split_fixed.yaml)):**
  - `5,547` images partitioned by video clip identity across `169` atomic sequences (`3,866` train, `852` val, `829` held-out test containing `1,170` instances: `600` `no_helmet`, `570` `helmet`).
  - SHA-256 and sequence-group audits confirm `0.0%` sequence leakage across splits ([`PHASE8_INDEPENDENT_EVALUATION_AUDIT.md`](../../results/phase8_independent_audit/PHASE8_INDEPENDENT_EVALUATION_AUDIT.md)).
- **Audited Mobile Street Expansion (`helmet-dataset-expansion-v2`):**
  - `1,841` handheld and street-level images (`1,285` train, `276` val, `280` held-out test containing `552` instances).
  - Audited across three human-inspected correction passes, refining `392` torso-bleeding bounding boxes across `318` images to tight cranial head-only boundaries (reducing max aspect ratio from `4.20` to `1.47` across all `3,733` boxes with `0` class inversions).
- **Combined Multi-Domain Corpus:**
  - Exact leak-free union: `7,388` images (`5,151` train, `1,128` val, `1,109` held-out test containing `1,722` instances).

### 1.2 Dual-Camera Real-Video Evaluation Corpus (`data/dual_camera_test_v1/`)
- **Scale & Sensors:** `25` roadside synchronized video pairs (`pair_001`–`pair_025`), `50` MP4 streams (`25` Android front cameras, `25` iOS/handheld back cameras), `15,241` total container packets/decoded corpus frames (`6,557` Front, `8,684` Back; `nb_frames == packet count == OpenCV sequential decode` across all `50 / 50` videos per [`PHASE12_FRAME_COUNT_RECONCILIATION.md`](PHASE12_FRAME_COUNT_RECONCILIATION.md)).
- **Stream Heterogeneity:** Three forensically verified variable-frame-rate (VFR) streams (`pair_001_front.mp4` at `59.72 fps` with 1 drop; `pair_013_back.mp4` at `40.15 fps`; `pair_014_back.mp4` at `48.47 fps`), four `30 fps` CFR back streams (`pair_015`–`018`), and `43` `60 fps` CFR streams.
- **Synchronization Strata (Phase 11.6 / 12C):** `21` sub-frame pairs ($\epsilon_{\text{sync}} \in [3.0, 10.0]\text{ ms}$) and `4` near-frame pairs ($\epsilon_{\text{sync}} \in [18.0, 38.0]\text{ ms}$, with `3` MEDIUM quality pairs `008, 019, 022`). Maximum labelled drift bound reaches `114.0 ms` in `pair_008`.
- **Canonical Reference Acceptance (`dual_camera_gt_v1.0_ai_assisted_audited` vs. `v1.1`):**
  - **Accepted Active Reference:** [`dual_camera_gt_v1.0_ai_assisted_audited`](../../data/dual_camera_test_v1/releases/dual_camera_gt_v1.0_ai_assisted_audited/README.md) (`AI_ASSISTED_AUDITED`, `human_verified: false`, `86` Pass-2/Pass-3/Stage-C correction operations, `70` global motorcycle events: `40` violations [`27` `DRIVER_NO_HELMET`, `10` `BOTH_NO_HELMET`, `3` `PILLION_NO_HELMET`], `6` `NO_VIOLATION`, `24` `INSUFFICIENT_VISUAL_EVIDENCE`; `29` front-only, `27` back-only, `10` linked-both-views, `4` ambiguous).
  - **Preserved Future Protocol (`v1.1`):** [`dual_camera_gt_v1.1_blind_human`](../../data/dual_camera_test_v1/releases/dual_camera_gt_v1.1_blind_human/README.md) remains untouched (`PENDING_HUMAN_REVIEW`, `0 / 92` completed human units, `0 / 1,564` non-empty blind response cells) per Principal Investigator directive; zero labels are fabricated or backfilled (see [`ANNOTATION_RELEASE_ACCEPTANCE_AND_PROVENANCE.md`](ANNOTATION_RELEASE_ACCEPTANCE_AND_PROVENANCE.md)).

---

## 2. Detector Training & Multi-Seed Reproducibility Protocol (Phases 7–9)

1. **Architecture & Initialization:** `YOLOv8n` initialized identically from official COCO-pretrained `yolov8n.pt` (`sha256: f59b3d83...`).
2. **Hyperparameters:** Input size $640 \times 640$, batch size $32$, optimizer $\text{AdamW}$ ($\beta_1=0.9$, $\beta_2=0.999$, weight decay $0.0005$), initial learning rate $0.001667$, momentum $0.9$, max epochs $50$, early-stopping patience $20$, deterministic mode enabled on NVIDIA Tesla T4 GPUs (Kaggle).
3. **Experimental Arms & Seeds:** Both `EXP-NEW` (mobile only) and `EXP-COMBINED` (multi-domain union) were trained across **three random seeds** ($\{42, 123, 2026\}$, `6` runs, `18` held-out test evaluations), and compared against the sequence-isolated CCTV baseline `EXP-002-FIXED`.
4. **Effect-Size & Stability Metrics:** Sample mean, sample standard deviation ($s$), min–max range, mean difference, and standardized Cohen's $d$. All weights are SHA-256 verified ([`exp_combined_checkpoints_sha256.json`](../../results/phase9_multiseed/exp_combined_checkpoints_sha256.json)).

---

## 3. Temporal Verification & Track-Level Redesign Protocols (Phase 10, TASK-04)

1. **Phase 10 Synthetic Suite (21 Runs):** Evaluates three aggregators (`simple_voting`, `confidence_weighted`, `ema`, sliding window $W=10$, $\alpha=0.35$) across seven standardized surveillance scenarios (`stable_helmet`, `stable_violation`, `fp_burst_on_compliant`, `short_occlusion`, `long_occlusion`, `marginal_jitter`, `multi_rider_crossing`) against confirmation latency, false alerts, state stability, and abstention rate.
2. **TASK-04 Genuine Observation Redesign ([`track_state_redesign.py`](../../scripts/reproducibility/track_state_redesign.py)):** Re-scores all `209` decided eligible GT rider tracks (plus the `85`-track single-observation $K=1$ subgroup) from frozen Stage J artifacts under `ema_final`, `last_coupled`, `majority`, `conf_weighted`, and `primary` accumulators, while verifying `0 / 85` regression differences in online alerts via [`event_regression_task04.py`](../../scripts/reproducibility/event_regression_task04.py).

---

## 4. Dual-Camera System Validation & Controlled Ablation Protocols (Phases 12C–12E, TASK-06–12)

To respect compute boundaries on local macOS M1 (`8 GB`), heavy full-video detection (`Stage J`) was frozen once and all downstream architectural improvements (`GAP-B/B1/B2/B3`, `GAP-C`, `TASK-10`, `TASK-11`, `TASK-12`) were evaluated via deterministic offline replay from frozen Stage J detections (`results/phase12_stagej/`) or bounded real-video frame decoding (`TASK-10`):

1. **Stage K Unified Audit ([`evaluate.py`](../../scripts/reproducibility/evaluate.py), [`bootstrap.py`](../../scripts/reproducibility/bootstrap.py)):** Evaluates detection, rider coupling (`6,727` spatially valid vs. `6,759` all coupled), IDF1/IDS (`tracking_head_basis`), association precision/recall/F1, event F1 (`2,000`-replicate pair-cluster bootstrap with `seed=20261109`), and explicitly declares non-computable families (`HOTA`, frame-level motorcycle AP, person AP, item-level rider association) rather than fabricating proxies.
2. **GAP-B Offline Audit (`TASK-06/06A`, [`evaluate_gapb_redesign.py`](../../scripts/reproducibility/evaluate_gapb_redesign.py)):** Verifies `24`-frame protected-anchor keyframe retention across all `50` streams (`100` motorcycle tracklets; `130` orphans / `86` riders in the frozen pre-GAP-C artifact `results/phase12_gapb_redesign/gapb_evaluation_results.json` vs. `68` orphans / `95` riders when replayed live against current `src/` GAP-C tracking), `4K -> 720p` coordinate recovery across `9` real PNG failure examples, `3840x2160` border false-positive elimination, and active-cue minimum gates (`min_active_cues_required=2`).
3. **GAP-C Tracking Ablation (`TASK-09/09A/09B`, [`evaluate_gapc_tracking.py`](../../scripts/reproducibility/evaluate_gapc_tracking.py)):** Controlled 5-arm ablation across all `25` pairs (`A0` Legacy Stage J $\to$ `A1` +Relative Kinematics $\to$ `A2` +Conflict Unlocking/Finite Coasting $\to$ `A3` +Longitudinal Role Consensus $\to$ `A4` +Orphan Lifecycle Gating).
4. **TASK-10 Bounded Real-Video Cue C (`evaluate_task10_real_video_cue_c.py`):** Decodes genuine frames across all 4 bounded opposing-view/convoy pairs (`pair_005, pair_008, pair_009, pair_019`; `8` streams decoded at both `720p` and native `4K`) comparing `C0_A4_no_cue_c`, `C1_A0_720p_kf`, `C2_A4_720p_kf`, `C3_A4_4k_native`, and `C4_A4_720p_none_dims_probe` for resolution invariance, self-similarity, and opposing-view discrimination.
5. **TASK-11 Fusion & Single-vs-Dual Ablation (`evaluate_task11_fusion.py`):** Evaluates `10` tracker $\times$ fusion arms (`A0`/`A4` $\times$ `{legacy, front_only, back_only, dual_fused}`, plus `A4_DUAL_FUSED_SPLIT` and `A4_DUAL_FUSED_MINOBS2`) under both **strict schema fixed-tIoU** (`N=70`) and **physical-pass deduplicated evaluation** (`70` passes, `40` violations, `46` decided passes), paired with **Wilson 95% score confidence intervals** and **two-sided exact McNemar binomial tests**.
6. **TASK-12 Boundary Audit ([`evaluate_task12_audit.py`](../../scripts/reproducibility/evaluate_task12_audit.py)):** Compares `compute_tiou(..., mode="legacy")` vs. `mode="fixed"` (assigning `tIoU=1.0` for identical point intervals `[t, t] == [t, t]` within `tol=1e-9`) and audits non-destruction of `v1.1`.
