# Integrated Dual-Camera Helmet System — RESEARCH LIMITATIONS (Stage L)

**Status: COMPLETE (2026-11-09).** This document collects every known limitation
of the dataset, the evaluation, and the integrated system, with the evidence for
each. Nothing here is softened: where a number elsewhere in the Stage L reports
could be misread, this document is the corrective.

## 1. The reference dataset is not human-verified

The frozen release `dual_camera_gt_v1.0_ai_assisted_audited` is labelled
**`AI_ASSISTED_AUDITED`** and `human_verified = false`. Every annotation was
produced by the project's own detectors and audited/corrected by AI passes
(Passes 1–3, Stages B–D). **No human being has verified this ground truth.**

This creates an unresolved tension with the project's own Phase 12B protocol
(`PHASE12B_DATA_ANNOTATION_PROTOCOL.md`), which prohibits treating unverified
model outputs as ground truth and requires human visual verification. The user's
mission explicitly directed AI-assisted annotation with AI re-audits; the
programme complied by labelling, not by upgrading the claim. Consequences:

- No result in this programme may be cited as accuracy against human judgement.
- Any publication or downstream use must either perform the human verification
  step or carry the AI-assisted label and this limitation with it.

## 2. The Stage K evaluation is bounded by construction circularity

The GT was constructed from Pass 1 detector outputs plus 86 audited corrections,
and Stage J bit-reproduces Pass 1. Therefore:

- Observation-level agreement on coupled units (head boxes, rider boxes, helmet
  states) is **perfect by construction** — the GT stores the detections verbatim.
  Those families certify *self-consistency of the coupling + correction +
  validity layers*, not detector accuracy.
- Bootstrap CIs for the by-construction-perfect metrics are degenerate [1, 1];
  they describe pair-resampling stability only.
- The genuinely independent error evidence in this programme is: (i) the visual
  audit samples (Passes 2/3, Stages B/D), and (ii) the aggregation-level
  divergences Stage K surfaced (helmet track-level abstention, association FP
  links, the pair_002 role attribution). Both are reported; neither supports a
  field-accuracy claim.

## 3. Residual annotation defects (measured, disclosed, partly unfixable in-layer)

- **Pass 3 fresh visual sample** (168 frames disjoint from Pass 2): residual
  frame error **39.3% blended** (66/168), of which **46/66 errors are rider
  subtrack bounding-box placement only** (states and the primary layer correct);
  the **detection/track-layer-only frame error rate is 11.9%**. Error profile is
  uniform across VFR/mixed, near-frame and other strata; front 50.0% vs back
  28.6% at frame level.
- **Full-coverage placement (Stage B):** of the 12,458 moto-subtrack rows, only
  49.4% are coupled to a same-frame detection; 46.0% are propagated-stale Kalman
  boxes and 4.6% off-frame — a **50.6% non-coupled share** (52.6% on the
  moto+orphan basis of 14,249 rows). Mechanism: 30 rider (sub)tracks rest on
  ≤2 coupled observations yet persist ≥30 rows (3,907 rows), flagged
  `LOW_EVIDENCE_SUBTRACK` in the validity layer.
- **The validity layer is a licence, not a cure.** `rider_frame_validity.csv`
  marks exactly **6,727 of 14,249** rider rows `spatial_use_valid`. Any spatial
  use (IoU, localization) of the other 7,522 rows — including **all 760 pair_002
  rows** (spatially unusable; its helmet *states* were retained because Stage B
  visually verified them) — is unsupported by the evidence.
- **Stage D fresh sweep residuals, disclosed and deliberately NOT fixed** in the
  frozen GT (SD-F1/F2/F3): 4 helmet-state errors at long-range/track-edge frames
  (3.3% of decided), 2 background-object boxes (1.6% placement), and one
  mostly-off-frame orphan box (pair_020 front_orphan_0002). Also 61 spatially-
  valid rows with minor (≤100 px) box overflow. No event outcome depends on any
  single one of these frames; fixing them inside a frozen release would have
  broken the audit chain, so they are carried in the release uncertainty log
  (178 rows total) instead.

## 4. Unresolved uncertainties (carried, never guessed)

- **pair_004 front orphans 0001 and 0003** span a genuine helmet-**donning**
  transition (helmet held beside a bare head → worn). No constant state is
  honestly assignable; 6 head observations are marked `occluded_unknown` with
  pre-Stage-C states preserved, and the packaged track states inside the donning
  windows are explicitly **not endorsed**. These orphan tracks do not feed any
  event.
- The Stage D sweep left 4 frames UNCERTAIN and Stage B's fresh sample 1; all
  are itemised in the respective verdict files rather than resolved by fiat.
- The release uncertainty log (`annotation_uncertainty.csv`, **178 rows**)
  is the canonical unresolved-items register for the frozen dataset.

## 5. Helmet track-level aggregation is weak

Observation-level helmet states are exact (by construction), but the pipeline's
**track-level** evidence aggregation agrees with the release on only **64.1%**
(134/209) of decided rider tracks: **73 tracks are pipeline-UNCERTAIN** because
their subtracks carry zero aggregated-confidence mass — the tracker's evidence
accumulation never fired for them (including every pair_002 subtrack). Only 2
hard cross-disagreements exist. Any consumer needing a per-rider verdict from
the raw pipeline (without the annotation layer's constant-state corrections)
will face this abstention rate.

## 6. Association abstains heavily; some families rest on tiny n

- Motorcycle association abstention **38.8%** (33/85); orphan-rider abstention
  **63.3%** (76/120). Abstentions are reported separately and never counted as
  correct — but a deployed system would experience them as non-decisions.
- Orphan association F1 = 1.000 rests on **exactly one** linked pair (n = 1) and
  carries no inferential weight.
- Rider association is available only as aggregate counts (Stage J emits no
  per-item rider results); no item-level claim exists.
- The links CSV's 70 event-derived moto rows vs 68 corrected association results
  (SB-006) is a counting-basis inconsistency in presentation, resolved
  documentarily in the release notes; the underlying bases (Stage A discrepancy
  D1: 12,458 track-frame / 13,062 GT `n_observations` / 6,759 direct head
  observations) must never be mixed.

## 7. Synchronization limits

- **Clock drift is reported, never compensated** (GAP-1): pairs 004/005/008/015
  carry +36/+70/−76/+74 ms of drift, so their stored ε_sync is a **lower bound**
  and end-of-window misalignment can reach ε_sync + |drift| (worst labelled
  bound: pair_008, 114 ms). All near-frame residuals are lower bounds; event
  timing on those pairs carries the same bound in wall-clock terms.
- **No exposure synchronization exists.** The cameras are independent consumer
  devices without genlock; alignment is container-timestamp level only (see the
  Synchronization Report §1 disclaimer).
- Three pairs are MEDIUM sync quality (008, 019, 022); three streams are
  forensically VFR (pair_001 front, pair_013/014 back) and four back streams run
  at 29.97 fps against ~60 fps fronts — handled by PTS indexing, but they are
  the strata where every residual above concentrates.
- No degraded single-camera mode exists (GAP-2): a missing camera fails the
  pair closed (`FAILED_PRECONDITION`) rather than producing partial output.

## 8. Metric families that do not exist (declared, not fabricated)

- **Person detection P/R/F1/AP** — the release contains no GT person entities.
- **Motorcycle frame-level box P/R and AP** — the release stores track-level
  motorcycle records (entry/peak/exit boxes), no per-frame GT boxes.
- **HOTA** — requires per-frame trajectories for motorcycle/person; unavailable
  in the release. A head-only variant would be non-standard and is not presented.
- **Rider association, item level** — no per-item predictions exist.
- The head-detection FP class "uncoupled" (3,024 of 3,046) cannot be split into
  detector false alarms vs coupling rejections from the frozen artifacts; the
  back camera's box-level precision (0.346) must be read with that caveat and
  with Phase 12D's known rear-view behaviour (back-camera motorcycle and helmet
  detections were frequently zero in smoke tests — rear views are intrinsically
  harder for these checkpoints).

## 9. Plates were never assessed

Plate visibility/text was **not assessed** in this programme: all 70 release
events carry `NOT_ASSESSED` and no plate text exists anywhere in the artifacts.
Nothing about plate readability can be claimed in either direction.

## 10. Engineering and delivery limitations

- **Repository tests:** the full suite stands at 140 passed / 3 failed; the 3
  failures are pre-existing environment issues on this Linux VM (missing
  `soundfile`, missing `easyocr`, an Apple-Silicon MPS test) — identical before
  and after every mission change, and unrelated to the pipeline.
- **Determinism scope:** detections/association artifacts are
  content-deterministic but not byte-deterministic — approved `to_dict()`
  payloads embed wall-clock latency fields and Phase 10 alert IDs embed a uuid4
  suffix (Stage G interpretation #9 / Stage H S-DET-1 scoping). Re-runs agree on
  content, not bytes.
- **Delivery:** the VM repo mirror has no `.git`. Mac sync and the git
  commit/push of all mission deliverables are **still pending** (Mac/GitHub HEAD
  `6aba04434af80cf0263de031a3c06488bebec29f`); until that happens the frozen
  artifacts exist only on this VM and in the Kaggle staging datasets.
- **Scope held throughout:** no training or fine-tuning was performed; no
  checkpoint was replaced; no OCR was implemented; the models are the existing
  Phase 9 / COCO checkpoints exactly as pinned. The system is a research
  prototype evaluated for internal consistency against an AI-assisted reference
  — it is not a validated enforcement or safety system.
