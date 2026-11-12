# Release Notes — dual_camera_gt_v1.0_ai_assisted_audited

Frozen at post-annotation **Stage E** (2026-11-09). Base: the Mission-1 corrected-view
package `data/dual_camera_test_v1/annotations/` (frozen, checksums 39/39, NOT modified by
this release) plus the Stage C correction layer (`work/stageC/`). Dataset label:
**`AI_ASSISTED_AUDITED`**; `human_verified = false`. Stage D's final audit verdict was
READY for this freeze; its §6 checklist is satisfied item by item in §3 below and its §7
disclosures are carried in §6.

## 1. Release identity
- Version: `dual_camera_gt_v1.0_ai_assisted_audited`
- Contents: 25 pairs / 50 videos / 15,241 frames; 70 events; 202 identity links;
  86 correction ops; 14,249-row rider-frame validity layer.
- The base package remains the frozen Mission-1 record. This release is a NEW directory;
  nothing in the base package, in Pass 1 (`work/pass1/`), in the original videos, or in
  Phase 11 metadata was modified.

## 2. The single event change (pair_020) — read this first
Incorporating SC-008 (EXCLUDE_FP_TRACK, pair_020 back_moto_0003 — a man riding a bicycle
loaded with orange cargo sacks, confirmed false positive over its whole 35-frame span in
Stage C and independently re-confirmed by Stage D) changes **exactly one** of the 70 events:

| | Base package | This release |
|---|---|---|
| `pair_020_moto_01` | tracks front_moto_0001 + back_moto_0003; AMBIGUOUS_ABSTAIN / AMBIGUOUS_HUMAN_UNCERTAIN; DRIVER_NO_HELMET | track front_moto_0001 alone; UNMATCHED / FRONT_VIEW_ONLY; DRIVER_NO_HELMET |
| Event total | 70 | **70 (unchanged)** |
| Type counts | 27 / 10 / 3 / 6 / 24 | **identical** |
| Link totals | LINKED 10 / FRONT_ONLY 28 / BACK_ONLY 27 / AMBIGUOUS 5 | LINKED 10 / FRONT_ONLY **29** / BACK_ONLY 27 / AMBIGUOUS **4** |
| Event IDs | — | **no renumbering anywhere** (the re-derived event takes the same ID by sort order) |

The violation outcome survives because back_moto_0003 has no rider subtracks and its
association result carried no rider associations: the NO_HELMET evidence came entirely from
front_moto_0001's slot-0 subtrack (131 observations). back_moto_0003 is recorded in
pair_020's `excluded_tracks` with correction_id SC-008. The links CSV changes by exactly
one row (the AMBIGUOUS_ABSTAIN motorcycle row is replaced by the UNMATCHED / FRONT_VIEW_ONLY
single-view row, using the same convention as the base package's own dropped-counterpart
rows); within pair_020 the row order follows the builder's semantics (single-view rows
follow retained-result rows). Association recount (result-derived basis): LINKED 10 /
UNMATCHED 34 / INSUFFICIENT_EVIDENCE 19 / AMBIGUOUS_ABSTAIN **4**; dropped motorcycle
results 17 -> **18**.

The events CSV `evidence_basis` text is updated on all 70 rows to cite 86 correction ops
(69 Mission-1 + 17 Stage C) and the Stage D re-verification; no other event field changes
on the other 69 events.

## 3. How each Stage C row (SC-001..SC-017) is embodied
| Row | Type | Embodiment in this release |
|---|---|---|
| SC-001 | ADD_RIDER_VALIDITY_LAYER | **Layer file**: `rider_frame_validity.csv` included (14,249 rows). No GT value altered. |
| SC-002 | FLAG_PAIR_SPATIALLY_UNUSABLE | **Validity layer**: all 760 pair_002 rows `spatial_use_valid=false`; states retained (see §6.2). Uncertainty row added. |
| SC-003 | SET_CONSTANT_HELMET_STATE | **GT edit**: pair_004 front_orphan_0005 track state -> HELMET, provenance MANUAL_CORRECTION, `correction_ids=[SC-003]`. |
| SC-004 | SET_CONSTANT_HELMET_STATE | **GT edit**: pair_004 front_orphan_0006 -> HELMET (already the modal packaged state; provenance/correction now recorded). |
| SC-005 | SET_CONSTANT_HELMET_STATE | **GT edit**: pair_004 front_orphan_0007 -> HELMET (as SC-004). |
| SC-006 | MARK_OBSERVATIONS_UNCERTAIN | **GT edit**: front_orphan_0001 head observations at f3, f4, f11, f12 -> `occluded_unknown`, originals in `state_before_stage_c`; track flagged NOT ENDORSED in transition windows (`stage_c_note`). |
| SC-007 | MARK_OBSERVATIONS_UNCERTAIN | **GT edit**: front_orphan_0003 head observations at f39, f40 -> `occluded_unknown`, originals preserved; track flagged NOT ENDORSED in the window. |
| SC-008 | EXCLUDE_FP_TRACK | **GT edit**: pair_020 back_moto_0003 removed from release GT, listed in `excluded_tracks`; produces the §2 event/link delta. |
| SC-009 | ADJUDICATED_NO_OP | **Documentary**: no data change, by design (pair_010 back_moto_0003 is a real motorcycle; support fraction alone never excludes). Recorded in the corrections log. |
| SC-010 | DOCUMENT_BASIS_NOTE | **Documentary**: the SB-006 reconciliation text, reproduced verbatim in §5. |
| SC-011 | INVALIDATE_RIDER_BOX_FRAME | **Validity layer**: pair_002 back_moto_0001_rider_s0 f287 `placement_class=INVALID`. |
| SC-012 | INVALIDATE_RIDER_BOX_FRAME | **Validity layer**: pair_019 back_moto_0002_rider_s1 f174 INVALID. |
| SC-013 | INVALIDATE_RIDER_BOX_FRAME | **Validity layer**: pair_019 back_moto_0002_rider_s0 f132 INVALID. |
| SC-014 | INVALIDATE_RIDER_BOX_FRAME | **Validity layer**: pair_019 back_moto_0002_rider_s0 f129 INVALID. |
| SC-015 | FLAG_OFF_FRAME_RIDER_ROWS | **Validity layer**: all 663 off-frame rider rows flagged OFF_FRAME / `spatial_use_valid=false`. |
| SC-016 | FLAG_LOW_EVIDENCE_SUBTRACKS | **Validity layer**: 30 (sub)tracks / 3,907 rows flagged LOW_EVIDENCE_SUBTRACK; caveat in §6.4 + uncertainty log. |
| SC-017 | SPOT_CHECK_RECORD | **Documentary**: neighbour spot-check record (evidence in `work/stageC/sb007_009_neighbor_spotcheck.csv`); its one data consequence (f129 invalidation) is SC-014. |

Mission-1 constant-state convention is followed for SC-003..005: per-frame
`head_observations` remain the raw detector records; the corrected state lives at track
level (`track_helmet_state` / `gt_helmet_state` / `state_provenance`).

## 4. Counting bases (Stage A discrepancy D1 — never mix)
Rider observations are counted on different bases in different artifacts, by definition:
**12,458** = rider track-frame rows, moto subtracks only (corrected view); **14,249** =
combined moto+orphan rider track-frame rows (validity-layer basis); **13,062** = sum of GT
subtrack `n_observations`; **6,759** = GT embedded direct head observations. Each artifact
states its basis; the numbers are not interchangeable and none is an error.

## 5. SB-006 reconciliation (verbatim, Stage C report §3 / corrections log SC-010)
> COUNTING BASIS (for Stage E release notes): cross_camera_identity_links.csv is event-derived - it carries one motorcycle row per global motorcycle event in dual_camera_gt_events.csv (70 rows). The corrected Phase 12E association recount is result-derived - it counts the association results that survive the corrected view (68 motorcycle results) after dropping results that reference FP-excluded tracks. The 2-row difference is exactly the event-derived UNMATCHED rows for pair_009 front_moto_0001 and pair_015 back_moto_0001, whose raw association results were dropped as FP-referencing. No link status is affected: the 10 LINKED_BOTH_VIEWS rows have both-side tracks present in the GT and dangling references are 0. The two bases must not be mixed: use 70 when counting events/links as packaged, 68 when counting corrected association outcomes.

## 6. Unresolved items this release DISCLOSES (not fixes) — from Stage D §7
1. **Propagated-box defect profile (intrinsic, flagged not fixed):** ~half of rider
   track-frame rows are non-coupled (validity layer: 6,834 PROPAGATED_STALE + 663 OFF_FRAME
   + 4 INVALID of 14,249). Spatial/statistical use is licensed only on the 6,727
   `spatial_use_valid` rows. No box was ever re-estimated in any stage.
2. **pair_002:** all 760 combined rows spatially unusable (states retained, visually
   verified in Stage B); the pair's event stays INSUFFICIENT_VISUAL_EVIDENCE.
3. **pair_004 donning transitions:** states for front_orphan_0001/0003 inside the transition
   windows are UNCERTAIN by adjudication (Stage C + Stage D concur); the packaged per-frame
   values inside the windows are explicitly **not endorsed** (preserved in
   `state_before_stage_c` for provenance).
4. **Low-evidence subtracks:** 30 (sub)tracks rest on <=2 coupled observations over >=30
   rows (3,907 rows flagged in the validity layer).
5. **Residual per-frame state error (Stage D measurement, SD-F1/F2):** 4 wrong helmet-state
   values remain in the GT head observations (pair_011 front f1, pair_015 front f0 orphan,
   pair_016 front f231 — NO_HELMET recorded, helmet worn; pair_023 front f6 — HELMET
   recorded spuriously), plus 2 residual background-object head boxes (pair_006 back f50
   foliage, pair_023 front f6 signboard). Measured fresh-sample rates: state 3.3% of
   decided, placement 1.6% (coupled stratum, 125 frames). These are disclosed residuals;
   the Stages C-D correction vocabulary does not include rewriting individual GT head
   observations, and no event outcome depends on any single one (events aggregate >=3
   coupled observations per rider). All are itemised in `annotation_uncertainty.csv`.
6. **pair_020 front_orphan_0002** mid-frame box mostly off-frame on background (SD-F3;
   orphan tracks do not feed events).
7. **61 spatially-valid rows** carry minor (<=100 px) frame-edge box overflow.
8. **Counting bases differ by artifact** — see §4.
9. **Mac sync and git commit/push** of the whole post-annotation work remain pending
   (outside this stage's gate; the VM mirror has no `.git`).

## 7. Phase 12B protocol tension (carried disclosure)
The project's own Phase 12B protocol prohibits unverified model-bootstrapped ground truth
and requires human visual verification. This mission was explicitly directed to proceed
AI-assisted with AI audit passes instead. The dataset is therefore labelled
**`AI_ASSISTED_AUDITED`** and must never be described as human-verified ground truth; no
human verification occurred at any stage (Passes 1-3, Stages A-E are all AI work).

## 8. Plates
Plate annotation was out of scope (no Phase 12F / no OCR). Every event carries
`plate_visibility_tier: NOT_ASSESSED` and `gt_plate_text_normalized: null`; 0 plates were
recorded and no plate text is asserted anywhere in this release.

## 9. Additive schema extensions (base schema/config are byte-identical copies)
- GT document header: `release` object; `corrections_applied` gains `stage_c_ops: 17`
  (total_ops 86).
- Five pair_004 orphan track records gain `correction_ids` and `stage_c_note` fields.
- Six pair_004 head observations gain `state_before_stage_c` and `stage_c_correction_id`.
- `annotation_corrections.csv` gains a `finding_id` column (Stage B finding refs for the
  Stage C rows; empty for Mission-1 rows).
- No base field was renamed, removed, or re-typed.

## 10. pair_progress_manifest.csv notes
Same columns as the base package. `corrections_applied` now counts all 86 ops by exact
pair field: pair_002 +2 (SC-002, SC-011), pair_004 +5 (SC-003..007), pair_010 +1 (SC-009),
pair_019 +3 (SC-012..014), pair_020 +1 (SC-008); the remaining 5 Stage C rows are global
(SC-001/010/015/016, pair=ALL) or multi-pair (SC-017) and are not attributed to a single
pair, so the column sums to 81 of 86 by design. pair_020: `excluded_tracks` 0 -> 1,
`moto_tracks_corrected` -1; `events_total` unchanged for every pair (70 total).
