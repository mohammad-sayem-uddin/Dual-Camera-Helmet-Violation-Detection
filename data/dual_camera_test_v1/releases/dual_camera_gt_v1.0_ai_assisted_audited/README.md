# dual_camera_gt_v1.0_ai_assisted_audited — Frozen Release

**Annotation label: `AI_ASSISTED_AUDITED` — this dataset is NOT human-verified ground truth.**
It was produced by an AI pipeline and audited by AI audit passes only. Do not cite it as
human-verified. The Phase 12B protocol's human-verification requirement was explicitly
overridden by the mission that produced this dataset; that tension is disclosed here and in
the audit reports (see RELEASE_NOTES.md).

## What this release is
The frozen release of the dual_camera_test_v1 annotations: all 25 synchronized front/back
camera pairs (50 videos, 15,241 decoded frames) — motorcycle tracks, rider subtracks with
helmet states, orphan (pedestrian / uncoupled rider) tracks, cross-camera identity links,
and per-event violation classifications — incorporating **all** correction layers:

- **69 Mission-1 correction ops** (Pass 2: 63 + Pass 3 addendum: 6), embodied in the base
  package `data/dual_camera_test_v1/annotations/` (which remains frozen and unchanged), plus
- **17 Stage C correction rows** (post-annotation Stage C, from the Stage B second
  independent audit), incorporated here at the Stage E freeze.

Total corrections log: **86 ops** (`annotation_corrections.csv`, provenance preserved per op).

## Headline counts (this release)
- Global motorcycle events: **70** — DRIVER_NO_HELMET 27, BOTH_NO_HELMET 10,
  PILLION_NO_HELMET 3, NO_VIOLATION 6, INSUFFICIENT_VISUAL_EVIDENCE 24.
- Cross-camera: LINKED_BOTH_VIEWS 10, FRONT_VIEW_ONLY **29**, BACK_VIEW_ONLY 27,
  AMBIGUOUS_HUMAN_UNCERTAIN **4**. (Front-only/ambiguous differ from the base package —
  exactly one event changed; see RELEASE_NOTES.md §2.)
- Identity links: 202 rows (motorcycle 70, rider 17, orphan_rider 115).
- Motorcycle tracks retained: front 42, back **40** (base package back 41; one further
  false-positive excluded here). Orphan tracks retained: front 71, back 54.
- Rider-frame validity layer: 14,249 rows; spatial use licensed on exactly **6,727** rows
  (`rider_frame_validity.csv`, `spatial_use_valid=true`).
- License plates: **not annotated**. Every event carries
  `plate_visibility_tier: NOT_ASSESSED` and `gt_plate_text_normalized: null`. No plate text
  is asserted anywhere.

## Read before using
- **RELEASE_NOTES.md** — what changed vs the base package, how every Stage C row is
  embodied, the SB-006 counting-basis reconciliation (verbatim), and every unresolved
  item this release discloses rather than fixes.
- Rider subtrack box coordinates are unreliable on non-coupled rows: ~half of rider
  track-frame rows are propagated/stale. Filter by `rider_frame_validity.csv`
  (`spatial_use_valid=true`) for any spatial/IoU use.
- Helmet-state residuals measured by the Stage D fresh sample (125 coupled frames):
  state error 3.3% of decided frames, placement error 1.6% (coupled stratum). Individual
  disclosed residual errors are listed in `annotation_uncertainty.csv`.
- Counting bases differ by artifact and must never be mixed: 12,458 moto rider track-frame
  rows / 14,249 combined (moto+orphan) rows / 13,062 GT n_observations / 6,759 GT embedded
  direct head observations.

## Layout
- `dual_camera_gt_v1.json` + `pairs/pair_NNN/pair_NNN_gt.json` — consolidated + per-pair
  annotations (Phase 12B hierarchical layout + per-camera track records).
- `dual_camera_gt_events.csv` — one row per global event (all-layers view).
- `cross_camera_identity_links.csv` — association outcomes per motorcycle/rider/orphan result.
- `rider_frame_validity.csv` — per rider track-frame spatial-validity layer (Stage C,
  Stage D-verified).
- `annotation_corrections.csv` — all 86 correction ops with provenance and rationale.
- `annotation_uncertainty.csv` — everything that remains uncertain or defective-by-disclosure.
- `frame_coverage_manifest.csv` / `video_inventory.csv` / `pair_progress_manifest.csv` —
  per-frame, per-video and per-pair accounting.
- `audit/` — Mission-1 audit CSVs (initial/correction/final), the Stage B second-audit
  findings CSV, and the Stage D final-audit results JSON.
- `annotation_schema.json` / `annotation_config.yaml` — schema and configuration of the
  base package (unchanged; additive release fields are documented in RELEASE_NOTES.md §8).
- `checksums/release_checksums.sha256` — SHA-256 for every release file except itself.

## Provenance chain
Pass 1 initial annotation (Phase 12D/12E pipeline, Kaggle T4) -> Pass 2 audit + 63 ops ->
Pass 3 final audit + 6 ops -> frozen base package -> Stage A completion verification
(15/15 gate) -> Stage B second independent audit (11 findings) -> Stage C correction layer
(17 rows) -> Stage D third/final audit (verdict READY; all Stage C claims reproduced or
visually confirmed) -> **Stage E freeze (this release)**. Reports:
`docs/phase12/PHASE12_ANNOTATION_AUDIT_1_REPORT.md`,
`docs/phase12/PHASE12_ANNOTATION_FINAL_AUDIT_REPORT.md`,
`docs/phase12/POST_ANNOTATION_STAGE_A_COMPLETION_VERIFICATION.md`,
`docs/phase12/POST_ANNOTATION_SECOND_INDEPENDENT_AUDIT.md`,
`docs/phase12/POST_ANNOTATION_CORRECTION_REPORT.md`,
`docs/phase12/POST_ANNOTATION_FINAL_AUDIT.md`,
`docs/phase12/POST_ANNOTATION_STAGE_E_FREEZE_REPORT.md`.
