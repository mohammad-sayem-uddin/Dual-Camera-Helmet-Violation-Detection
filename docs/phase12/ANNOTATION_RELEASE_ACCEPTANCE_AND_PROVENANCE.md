# Annotation Release Acceptance & Provenance Audit

**Project:** Dual-Camera Context-Aware Motorcycle Helmet Violation Detection  
**Audit Date:** 2026-11-13  
**Acceptance Audit Basis:** `e68b742` (`main`, prior to final engineering/documentation completion commits)  
**Decision Authority:** Principal Investigator / Project Owner Priority Directive  

---

## 1. Executive Acceptance Decision

Per project owner directive, blind multi-annotator human review (`TASK-12` human execution phase) is halted without recruiting external reviewers or completing human annotation cells. No human labels are fabricated, inferred, or backfilled.

Accordingly, the project adopts the following canonical release policy:

1. **Accepted Active Evaluation Reference:**
   - **Release:** [`data/dual_camera_test_v1/releases/dual_camera_gt_v1.0_ai_assisted_audited/`](../../data/dual_camera_test_v1/releases/dual_camera_gt_v1.0_ai_assisted_audited/README.md) (`schema_version: 1.0.0`)
   - **Provenance Label:** `AI_ASSISTED_AUDITED` (`human_verified: false`)
   - **Status:** Frozen, SHA-256 verified (`43/43` release files, `39/39` base annotation files), and accepted as the project's **sole active evaluation reference** for Stage K, `TASK-04`, `TASK-06/06A`, `TASK-09A/09B`, `TASK-10`, `TASK-11`, and `TASK-12` offline benchmarking.
2. **Retained Unexecuted Future Protocol & Workqueue Template:**
   - **Release:** [`data/dual_camera_test_v1/releases/dual_camera_gt_v1.1_blind_human/`](../../data/dual_camera_test_v1/releases/dual_camera_gt_v1.1_blind_human/README.md) (`schema_version: 1.1.0-pending-review`)
   - **Provenance Label:** `PENDING_HUMAN_REVIEW` (`human_verified: false`, `executed_human_units: 0`, `fabricated_labels_present: false`)
   - **Status:** Preserved intact and unmodified as a non-destructive schema, stratification specification, and blind review infrastructure template for any future human-labeling continuation. It must **never** be cited as completed human ground truth.

---

## 2. Cryptographic & Structural Provenance Verification

### 2.1 `dual_camera_gt_v1.0_ai_assisted_audited` (Accepted Reference)

All checksum manifests verified with exit code `0` on `2026-11-13`:

| Artifact Scope | Manifest Path | Verified Count | Status |
|---|---|---|---|
| Base Corrected Annotations | [`annotation_checksums.sha256`](../../data/dual_camera_test_v1/annotations/checksums/annotation_checksums.sha256) | `39 / 39` | `OK` |
| Stage E Frozen Release `v1.0` | [`release_checksums.sha256`](../../data/dual_camera_test_v1/releases/dual_camera_gt_v1.0_ai_assisted_audited/checksums/release_checksums.sha256) | `43 / 43` | `OK` |
| Frozen Stage K Reference Outputs | [`checksums.sha256`](../../results/phase12_integrated/checksums.sha256) | `8 / 8` | `OK` |

Verified metadata and record counts inside `dual_camera_gt_v1.0_ai_assisted_audited`:

- **Corpus Scale:** `25` dual-camera video pairs (`pair_001`..`pair_025`), `50` synchronized MP4 streams, `15,241` total decoded frames (`6,557` Front / `8,684` Back in `frame_coverage_manifest.csv`).
- **Models Recorded in [`dual_camera_gt_v1.json`](../../data/dual_camera_test_v1/releases/dual_camera_gt_v1.0_ai_assisted_audited/dual_camera_gt_v1.json):**
  - `vehicle_person`: `yolov8n.pt` (`sha256: f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`, classes `{0: "person", 3: "motorcycle"}`)
  - `helmet`: `results/phase9_multiseed/exp_combined/seed_42/weights/best.pt` (`sha256: 0929a882667e0f7e2b631071ab218a2415cc1b816cdede7fb6812543d840b0e5`, classes `{0: "no_helmet", 1: "helmet"}`)
  - Both live repository model files (`yolov8n.pt`, `results/phase9_multiseed/exp_combined/seed_42/weights/best.pt`, and `results/exp_combined/weights/best.pt`) match these exact SHA-256 digests.
- **Provenance & Correction History (`corrections_applied.total_ops = 86`):**
  - `annotated_by`: `"ai_assisted_pipeline_pass1_pass2_pass3_audited"`
  - `pass2_ops`: `63`, `pass3_addendum_ops`: `6`, `stage_c_ops`: `17` (`total_ops = 86` rows in `annotation_corrections.csv`).
  - Correction breakdown by `correction_type`: `EXCLUDE_FP_TRACK` (`23`), `CLIP_OFF_FRAME_BOX` (`20`), `KEEP_FRAGMENT_TRACK` (`19`), `SET_CONSTANT_HELMET_STATE` (`11`), `INVALIDATE_RIDER_BOX_FRAME` (`4`), `MARK_OBSERVATIONS_UNCERTAIN` (`2`), and `1` each of `ADD_RIDER_VALIDITY_LAYER`, `FLAG_PAIR_SPATIALLY_UNUSABLE`, `ADJUDICATED_NO_OP`, `DOCUMENT_BASIS_NOTE`, `FLAG_OFF_FRAME_RIDER_ROWS`, `FLAG_LOW_EVIDENCE_SUBTRACKS`, `SPOT_CHECK_RECORD`.
- **Global Motorcycle Events (`dual_camera_gt_events.csv`, `n = 70`):**
  - **By `event_type`:** `DRIVER_NO_HELMET` (`27`), `BOTH_NO_HELMET` (`10`), `PILLION_NO_HELMET` (`3`), `NO_VIOLATION` (`6`), `INSUFFICIENT_VISUAL_EVIDENCE` (`24`). Thus `40` violation events, `6` compliant (`NO_VIOLATION`), `24` insufficient visual evidence.
  - **By `cross_camera_link_status`:** `FRONT_VIEW_ONLY` (`29`), `BACK_VIEW_ONLY` (`27`), `LINKED_BOTH_VIEWS` (`10`), `AMBIGUOUS_HUMAN_UNCERTAIN` (`4`).
  - **License Plates (`global_motorcycle_events[*].plate_ground_truth`):** `70 / 70` entries carry `plate_visibility_tier: 'NOT_ASSESSED'` with `gt_plate_text_normalized: None` (ALPR/OCR is explicitly out of scope).
- **Cross-Camera Identity Links (`cross_camera_identity_links.csv`, `n = 202`):**
  - **By `kind`:** `motorcycle` (`70`), `orphan_rider` (`115`), `rider` (`17`).
  - **By `outcome`:** `UNMATCHED` (`83`), `INSUFFICIENT_EVIDENCE` (`80`), `AMBIGUOUS_ABSTAIN` (`24`), `LINKED` (`15`).
- **Frame Validity & Uncertainty Layers:**
  - `rider_frame_validity.csv`: `14,249` rows total — `6,727` (`47.2%`) `spatial_use_valid=true`, `7,522` (`52.8%`) `spatial_use_valid=false`.
  - `annotation_uncertainty.csv`: `178` uncertainty rows.

### 2.2 `dual_camera_gt_v1.1_blind_human` (Preserved Unexecuted Template)

Directly audited file inventory and SHA-256 hashes:

| File | Size (B) | SHA-256 | Verified State |
|---|---|---|---|
| [`README.md`](../../data/dual_camera_test_v1/releases/dual_camera_gt_v1.1_blind_human/README.md) | `2488` | `5ba00de48c38f384b388df2bda0154e37fef60fade1b15a14151c4f8260b8f1a` | Protocol overview (`PENDING_HUMAN_REVIEW`) |
| [`blind_annotation_workqueue.csv`](../../data/dual_camera_test_v1/releases/dual_camera_gt_v1.1_blind_human/blind_annotation_workqueue.csv) | `22078` | `27c18280b3cf781bc043a07cc29c095b198240c131b3c931833d1888ea4b4911` | `92` rows, `17` blind response cols, `0 / 1,564` filled cells, `92/92 PENDING_HUMAN_REVIEW` |
| [`release_metadata.json`](../../data/dual_camera_test_v1/releases/dual_camera_gt_v1.1_blind_human/release_metadata.json) | `1280` | `3e270c9ae8901a0d7da1c73e2d7761cbbf19a6258e3f9f14ac35947717ecb45d` | `human_verified: false`, `executed_human_units: 0`, `pending_human_units: 92` |
| [`schema.json`](../../data/dual_camera_test_v1/releases/dual_camera_gt_v1.1_blind_human/schema.json) | `2569` | `be2d59c6854156abe2d38f925804063f7b2ecc14eb9de36bc79412f89d5f342d` | JSON Schema `dual_camera_gt_v1_1_blind_human.schema.json` |
| [`review_workspace/README.md`](../../data/dual_camera_test_v1/releases/dual_camera_gt_v1.1_blind_human/review_workspace/README.md) | `1543` | `142c6f06e167fb1e293b1f11f82aac04e250c9a1eea24d953830852547e6e344` | Sole file in `review_workspace/` (no SQLite DB or reviewer JSONs written) |

Workqueue stratification in `blind_annotation_workqueue.csv` (`92` items across `25` pairs):
- **By `source_origin`:** `PHYSICAL_TRANSIT_CORPUS` (`70`), `GAPC_UNLINKED_OR_SPLIT_CANDIDATE` (`22`).
- **By `priority_stratum`:** `STRATUM_1_GAPC_PILLION_OR_CONFLICT` (`46`), `STRATUM_2_UNCERTAIN_AMBIGUOUS_OR_DELTA` (`33`), `STRATUM_3_BASELINE_AND_PLATE_AUDIT` (`13`).

---

## 3. Temporary Review Dashboard Disposition

The local blind-review web server ([`src/review_dashboard/`](../../src/review_dashboard/server.py), launched via [`scripts/start_gt_review_dashboard.sh`](../../scripts/start_gt_review_dashboard.sh) with automated test coverage in [`tests/test_phase12_review_dashboard.py`](../../tests/test_phase12_review_dashboard.py)) is retained as clean, non-destructive tooling:

- Zero runtime writes have been committed into `review_workspace/` (only `README.md` is present).
- All checked-in `v1.0` and `v1.1` hashes remain undisturbed.
- Retaining the dashboard keeps future human-labeling reproducibility intact without further manual annotation investment during this project completion phase.

---

## 4. Mandatory Methodological Disclosures & Scientific Boundaries

Every technical report, README table, and academic manuscript relying on `dual_camera_gt_v1.0_ai_assisted_audited` **must** explicitly state these four constraints:

1. **AI-Assisted Circularity Risk:** `v1.0` events and track boundaries originated from the Phase 12D/12E pipeline (`yolov8n.pt` + Phase 9 `seed_42/best.pt`), subsequently audited through Pass 2 / Pass 3 / Stage C corrections (`86` operations). Consequently, evaluation scores on `v1.0` quantify **architectural consistency, cross-camera association/fusion logic, and regression deltas**, not zero-bias generalization against independent human ground truth.
2. **Absence of Blind Human Consensus:** Inter-rater reliability (Cohen's kappa targets in `v1.1` `release_metadata.json`: `>= 0.85` binary violation, `>= 0.85` rider helmet, `>= 0.90` cross-camera link) remains **unmeasured** because `executed_human_units = 0`.
3. **High Unresolved / Single-View Rate in Reference:** `24 / 70` (`34.3%`) of `v1.0` events carry `INSUFFICIENT_VISUAL_EVIDENCE`, and only `10 / 70` (`14.3%`) physical transits are `LINKED_BOTH_VIEWS` in `v1.0` (`29` front-only, `27` back-only, `4` ambiguous), constraining statistical power for cross-view fusion comparisons.
4. **No License Plate Validation:** `70 / 70` events have `plate_visibility_tier = NOT_ASSESSED`; license plate detection or OCR claims cannot be made.

---

## 5. Acceptance Sign-Off Summary

| Criterion | Verified Result |
|---|---|
| `v1.0` cryptographic integrity | `39/39` annotation + `43/43` release + `8/8` Stage K checksums `OK` |
| `v1.1` zero-fabrication invariant | `0 / 1,564` blind response cells filled; `92 / 92 PENDING_HUMAN_REVIEW` |
| Accepted benchmark reference | `dual_camera_gt_v1.0_ai_assisted_audited` (`70` events, `25` pairs, `15,241` frames) |
| Legacy `v1.0` immutability | Enforced; no script writes into `data/dual_camera_test_v1/` |
