# dual_camera_test_v1 — Final Annotation Package (Corrected View)

**Annotation label: `AI_ASSISTED_AUDITED` — this dataset is NOT human-verified ground truth.**
It was produced by an AI pipeline and audited by AI audit passes (see below). Do not cite it as
human-verified. The Phase 12B protocol's human-verification requirement was explicitly overridden
by the mission that produced this package; that tension is disclosed here and in the audit reports.

## What this is
Complete annotations for all 25 synchronized front/back camera pairs (50 videos, 15,241
decoded frames) of the dual-camera helmet-violation test set: motorcycle tracks, rider subtracks
with helmet states, orphan (pedestrian / uncoupled rider) tracks, cross-camera identity links,
and per-event violation classifications — in the **corrected view**: Pass 1 initial annotations
plus 69 correction operations
(63 from Pass 2 + 6 Pass 3 addendum), applied as a view. The Pass 1 release itself is immutable
and is not contained in this package.

## How it was produced
- **Pass 1 (initial annotation):** approved Phase 12D per-camera detection+tracking and Phase 12E
  cross-camera association code, run on Kaggle T4. Models: `yolov8n.pt` (COCO person/motorcycle,
  sha256 `f59b3d83…`) and the Phase 9 helmet model `best.pt` (no_helmet/helmet head detections,
  sha256 `0929a882…`). Sequential decoding with original PTS only.
- **Pass 2 (audit + corrections):** programmatic audit (220 findings), visual audit of 258 sampled
  frames (161 correct / 9 errors / 88 uncertain), full-resolution resolution of all 88 uncertain
  frames (all 73 apparent helmet-state flips were detector noise; 8 subtracks carry one constant
  audited state each), 63 correction ops. Report: `docs/phase12/PHASE12_ANNOTATION_AUDIT_1_REPORT.md`.
- **Pass 3 (final independent audit):** source-video SHA-256 integrity 50/50, corrected-view
  validation 20/20 checks, correction spot-verification 24/24 confirmed, fresh visual sample of
  168 previously unreviewed frames. It found 6 further false-positive motorcycle tracks, excluded
  by the 6 addendum ops in this package (69 ops total).
  Report: `docs/phase12/PHASE12_ANNOTATION_FINAL_AUDIT_REPORT.md`.

## Measured quality (Pass 3 fresh sample, 168 frames — read before using)
- Correct 94 / Error 66 / Uncertain 8 → **residual frame error rate 39.3%**
  (41.3% of decided frames).
- **46 of the 66 errors are rider-subtrack bounding-box PLACEMENT defects only** (stale Kalman
  boxes, boxes floating on background, degenerate/off-frame boxes): in those frames the
  motorcycle, the detections and the helmet-state values are correct, but the rider subtrack
  box coordinates are unreliable. **Do not use rider subtrack box coordinates for spatial
  IoU evaluation without filtering**; the sample-identified defective frames are listed in
  `annotation_uncertainty.csv`.
- Detection/track-layer frame error rate is **11.9%** (20/168, false-positive detections).
- Helmet-state VALUES were verified correct at every coupled check in Pass 3, including all
  8 constant-state corrections.
- Coverage is 100% (15,241/15,241 frames) — coverage is not correctness.

## Headline counts (corrected view)
- Global motorcycle events: 70 — violations: DRIVER_NO_HELMET 27,
  BOTH_NO_HELMET 10, PILLION_NO_HELMET 3;
  NO_VIOLATION 6; INSUFFICIENT_VISUAL_EVIDENCE 24.
- Cross-camera: LINKED_BOTH_VIEWS 10, FRONT_VIEW_ONLY 28,
  BACK_VIEW_ONLY 27, AMBIGUOUS 5.
- Rider helmet observations (per-frame derived, corrected): HELMET 1,825,
  NO_HELMET 5,686, UNCERTAIN 4,947.
- License plates: **not annotated** (OCR was out of scope). Every event carries
  `plate_visibility_tier: NOT_ASSESSED` and `gt_plate_text_normalized: null`. No plate text is asserted anywhere.

## Package layout
- `dual_camera_gt_v1.json` — consolidated corrected annotations (Phase 12B hierarchical layout + per-camera track records).
- `pairs/pair_NNN/pair_NNN_gt.json` — the same content per pair.
- `dual_camera_gt_events.csv` — one row per global event with violation classification.
- `cross_camera_identity_links.csv` — corrected association outcomes per motorcycle/rider/orphan result.
- `frame_coverage_manifest.csv` — all 15,241 frames with PTS and per-frame detection counts.
- `video_inventory.csv` / `pair_progress_manifest.csv` — per-video and per-pair accounting.
- `annotation_corrections.csv` — all 69 correction ops with provenance and rationale.
- `annotation_uncertainty.csv` — everything that remains uncertain or defective-by-disclosure.
- `audit/` — distilled Pass 2 initial findings, Pass 2/3 correction-audit verdicts, Pass 3 final-audit results.
- `annotation_schema.json` / `annotation_config.yaml` — schema and the exact configuration/semantics used.
- `checksums/annotation_checksums.sha256` — SHA-256 for every package file except the checksum file itself.

## Limitations
1. AI-assisted and AI-audited only; no human verification occurred.
2. Rider subtrack box placement is the dominant residual defect (see measured rates above).
3. Back-camera detections are frequently absent for rear views in the source pipeline; audits
   treated detector silence as absence of evidence, never as evidence of absence.
4. Several Phase 12B fields were not recorded by the Pass 1 pipeline and are null
   (occlusion level, headgear note, per-view visibility, plate data).
5. Residual error rates come from a 168-frame stratified sample, not a full re-inspection.
