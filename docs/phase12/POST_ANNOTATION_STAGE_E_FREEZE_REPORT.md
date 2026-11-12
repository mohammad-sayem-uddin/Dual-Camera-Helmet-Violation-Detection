# Post-Annotation Mission — Stage E: Freeze Report

**Date:** 2026-11-09 · **Stage:** E of A–L · **Release:** `dual_camera_gt_v1.0_ai_assisted_audited`
**Location:** `data/dual_camera_test_v1/releases/dual_camera_gt_v1.0_ai_assisted_audited/` (44 files, 8.2 MB)
**Builder:** `work/stageE/build_release.py` · **Independent validator:** `work/stageE/validate_release.py`
(46/46 PASS, `work/stageE/validation_results.json`) · **Progress:** `work/stageE/STAGE_E_PROGRESS.md`

> Dataset status remains **`AI_ASSISTED_AUDITED`** — AI-generated, AI-audited, AI-corrected.
> `human_verified = false`. Nothing in this stage makes it human-verified ground truth, and
> the release says so in its README, release notes, and GT header.

---

## 1. What was built

The frozen release = the frozen Mission-1 package (base, byte-identical starting point)
**plus** the full Stage C correction layer incorporated, per the Stage D §6 freeze
checklist — every item satisfied:

- [x] Frozen package as base (checksums 39/39 before AND after this stage; 40 files;
      the package directory was never written to).
- [x] All 69 Mission-1 ops — already embodied in the base; provenance records carried
      into the release corrections log unchanged.
- [x] All 17 Stage C rows incorporated (§3), SC-008 producing exactly the Stage D §4 delta.
- [x] Rider-frame validity layer included (`rider_frame_validity.csv`, 14,249 rows;
      COUPLED 6,748 / PROPAGATED_STALE 6,834 / OFF_FRAME 663 / INVALID 4;
      `spatial_use_valid=true` on exactly 6,727 rows; pair_002 spatially unusable;
      30 low-evidence (sub)tracks flagged).
- [x] SB-006 reconciliation text **verbatim** in RELEASE_NOTES.md §5 — verified
      character-identical in both sources (Stage C report §3 blockquote and corrections
      log SC-010) before embedding.
- [x] Events/links rebuilt from the all-layers view (View B), event IDs preserved
      (no renumbering), release note recording the single pair_020 change.
- [x] Plate fields NOT_ASSESSED throughout (0 plates recorded; no plate text anywhere).
- [x] `AI_ASSISTED_AUDITED` label + Phase 12B protocol-tension disclosure carried in
      README.md, RELEASE_NOTES.md (§7), and the GT `provenance_note`.

**Release inventory (44 files):** consolidated GT + 25 per-pair GT files, events CSV,
links CSV, corrections CSV (86 ops), uncertainty CSV (178 rows), validity layer,
frame-coverage manifest, video inventory, pair progress manifest, schema + config
(byte-identical copies), audit/ (3 Mission-1 CSVs + Stage B findings CSV + Stage D
results JSON), README.md, RELEASE_NOTES.md, checksums/release_checksums.sha256
(43 entries, **43/43 verified**).

## 2. Build method — derivation proven faithful before anything was changed

The release was NOT hand-edited. Procedure (`build_release.py`):

1. **View A faithfulness proof.** The package builder's event/link logic
   (`work/package/build_package.py` Part 2) was replicated and run over the *frozen*
   GT with only the 69 Mission-1 exclusions. Result: the derived events CSV and links
   CSV are **byte-identical** to the packaged files, and the derived per-pair event
   records equal the packaged records for all 25 pairs. Only after this proof were
   the same functions trusted to produce View B.
2. **Stage C GT transformations** (§3) applied to a deep copy of the GT.
3. **View B derivation** with the exclusion set extended by SC-008; all 20 expectation
   checks taken from Stage D's `step4_results.json` passed exactly (§4).
4. Files written; checksums generated and verified; frozen package re-verified (39/39).

## 3. How each Stage C row was embodied

Full table in the release's RELEASE_NOTES.md §3. Summary by disposition:

- **GT edits (6 rows):**
  - SC-003/004/005 — `SET_CONSTANT_HELMET_STATE` on pair_004 front orphans
    0005/0006/0007 → track state HELMET, `gt_helmet_state=helmet`,
    `state_provenance=MANUAL_CORRECTION` (the package's own constant-op convention),
    `correction_ids` + `stage_c_note` recorded. (0005's packaged state was NO_HELMET —
    superseded; 0006/0007 were already HELMET — provenance now recorded.) Per-frame
    `head_observations` remain the raw detector records, exactly as Mission-1
    constants left them; the corrected state lives at track level.
  - SC-006/007 — `MARK_OBSERVATIONS_UNCERTAIN` on the genuine donning-transition
    tracks: head observations at front_orphan_0001 f3, f4, f11, f12 (originals:
    helmet, helmet, no_helmet, no_helmet) and front_orphan_0003 f39, f40 (helmet,
    helmet) are set to `occluded_unknown` with originals preserved in
    `state_before_stage_c` and `stage_c_correction_id` recorded. Packaged track-level
    states retained but explicitly **NOT ENDORSED** inside the transition windows
    (`stage_c_note` + uncertainty log rows).
  - SC-008 — `EXCLUDE_FP_TRACK` pair_020 back_moto_0003: removed from the release
    GT's back motorcycle tracks; recorded in pair_020 `excluded_tracks` with
    `correction_ids=[SC-008]` in the package's own entry format.
- **Validity-layer rows (6):** SC-001 (layer file included), SC-002 (pair_002 flag),
  SC-011..014 (the 4 INVALID frames), SC-015 (663 OFF_FRAME flags), SC-016
  (LOW_EVIDENCE flags) — all embodied by `rider_frame_validity.csv`, byte-identical
  to the Stage D-verified Stage C file. No GT value altered by these rows.
- **Documentary / no-op rows (3):** SC-009 (adjudicated no-op, pair_010 back_moto_0003
  retained), SC-010 (SB-006 basis note → RELEASE_NOTES §5), SC-017 (spot-check record;
  its one data consequence is SC-014). Recorded in the corrections log with rationale;
  no data change, by design.

`annotation_corrections.csv` = the 69 package rows (unchanged, in order) + the 17
Stage C rows, with one additive column `finding_id`. `annotation_uncertainty.csv` =
the 161 package rows (unchanged) + 17 Stage C/D rows: 2 donning-transition, 4 residual
state errors (SD-F1), 2 background-object boxes (SD-F2), 1 pair_020 orphan placement
defect (SD-F3), 4 Stage D fresh-sample uncertain frames, the propagated-box spatial-use
restriction, the pair_002 states-retained flag, the low-evidence disclosure, and the
61-row minor-overflow disclosure (5 rows sourced STAGE_C_CORRECTION, 12 STAGE_D_FINAL_AUDIT).

`pair_progress_manifest.csv` keeps the package's columns; only release-affected numbers
changed (documented in RELEASE_NOTES §10): per-pair `corrections_applied` recounts all
86 ops by exact pair field (pair_002 +2, pair_004 +5, pair_010 +1, pair_019 +3,
pair_020 +1; the 5 global/multi-pair Stage C rows are unattributed by design, so the
column sums to 81 of 86); pair_020 `excluded_tracks` 0→1, `moto_tracks_corrected` −1;
`events_total` unchanged for every pair.

## 4. Exact release counts (all independently recounted by the validator)

| Quantity | Value | Stage D expectation |
|---|---|---|
| Events | **70** | 70 ✓ |
| Event types | DRIVER_NO_HELMET 27 / BOTH_NO_HELMET 10 / PILLION_NO_HELMET 3 / NO_VIOLATION 6 / INSUFFICIENT 24 | identical ✓ |
| Link totals (events) | LINKED_BOTH_VIEWS 10 / FRONT_VIEW_ONLY **29** / BACK_VIEW_ONLY 27 / AMBIGUOUS **4** | ✓ |
| Links CSV rows | **202** (motorcycle 70 / rider 17 / orphan_rider 115) | 202 ✓ |
| Links delta vs package | exactly 1 row swapped (pair_020 AMBIGUOUS_ABSTAIN row → UNMATCHED/FRONT_VIEW_ONLY single-view row) | ✓ |
| Association recount (moto) | LINKED 10 / UNMATCHED 34 / INSUFFICIENT 19 / ABSTAIN **4**; dropped moto **18** / orphan 5 | ✓ |
| Changed event | `pair_020_moto_01` only: front_moto_0001 alone, UNMATCHED / FRONT_VIEW_ONLY, DRIVER_NO_HELMET, driver NO_HELMET, window [3.32, 5.319511] — **same ID, zero renumbering** | ✓ |
| Other 69 events | identical to package on every field except the updated `evidence_basis` text (86 ops) | — |
| Moto tracks retained | front 42 / back **40** (package back 41 − SC-008) | — |
| Orphan tracks retained | front 71 / back 54 (unchanged) | — |
| Excluded tracks (GT) | 23 (22 Mission-1 + SC-008) | — |
| Corrections log | **86 ops** (63 Pass 2 + 6 Pass 3 + 17 Stage C) | — |
| Uncertainty log | **178 rows** (161 + 17) | — |
| Validity layer | 14,249 rows; spatial-valid exactly 6,727 | ✓ (Stage D-verified) |
| Plates | 0 recorded; NOT_ASSESSED everywhere | ✓ |

**Deviations from Stage D §4 expectations: NONE.** All 20 build-time expectation checks
and all 46 independent validation checks pass.

## 5. Validation (independent script, fresh reads of the built release)

`work/stageE/validate_release.py` — **46/46 PASS**, covering: release checksums +
inventory; GT parse/schema sanity (label, `human_verified=false`, plates NOT_ASSESSED
on every event); per-pair files == consolidated; events/links vs Stage D View B and vs
the package (incl. no-renumbering and the single-swap links delta); track inventory;
the pair_004 edits (constants + the 6 marked observations with exact original values);
**diff discipline** — outside pair_004/pair_020 the release GT's camera annotations and
excluded tracks are deep-equal to the package, pair_004 differs in exactly the 5 orphan
records, pair_020 in exactly the 1 event record; corrections/uncertainty composition
and prefix-preservation; validity-layer counts + byte-identity; byte-identical copies;
manifest recount; referential integrity (dangling = 0); no live reference to
pair_020 back_moto_0003; frozen package still 39/39 + 40 files; Pass 1 aggregate
SHA-256 unchanged; SB-006 verbatim present in RELEASE_NOTES.

**Honest process note:** the validator's first run failed 2 of 46 checks. Both failures
were defects in the validator's check definitions, proven so with data before fixing:
(a) the dangling-reference check used a per-kind identity basis, but Phase 12E
`orphan_results` legitimately pair an orphan track on one camera with a coupled moto
rider subtrack on the other (11 such rows, byte-identical in the frozen package) —
on Stage D §5's own union basis (tracks ∪ subtracks ∪ orphans), dangling = 0 in both
the package and the release; (b) the pair_020 exclusion check was not pair-scoped for
events — pairs 006/010 have their own legitimate back_moto_0003 tracks; scoped to
pair_020, live references = 0. The release itself was not changed in response; only
the check definitions were corrected, and the corrected run passes 46/46.

## 6. What was NOT changed

- The frozen base package (39/39 checksums verified after the build), Pass 1
  (301 files; aggregate SHA-256 `585c1c45…` identical before/after), the original
  videos, and all Phase 11 metadata: untouched.
- No bounding box was re-estimated, moved, or redrawn anywhere — the Stage D residual
  box/state defects are **disclosed** (RELEASE_NOTES §6, uncertainty log), not fixed.
- No event outcome changed except the single documented pair_020 link-status change;
  its violation type is unchanged.

## 7. Carry-forwards

- Stage D §7 disclosures all live in RELEASE_NOTES.md §6 and the uncertainty log —
  they are properties of the frozen release, disclosed not fixed (propagated-box
  profile with spatial licensing on 6,727 rows; pair_002 spatially unusable with
  states retained; donning-transition UNCERTAIN residue; 30 low-evidence subtracks;
  SD-F1/F2/F3 residuals; 61 minor-overflow rows; counting bases).
- Stage K must filter rider-spatial metrics by `spatial_use_valid` and carry the
  SB-011 low-observation caveat (noted in RELEASE_NOTES).
- Mac sync + git commit/push of the whole post-annotation work remain pending
  (outside this stage's gate; the VM mirror has no `.git`).

**Stage E: COMPLETE.** Release `dual_camera_gt_v1.0_ai_assisted_audited` is frozen at
`data/dual_camera_test_v1/releases/dual_camera_gt_v1.0_ai_assisted_audited/` with
checksums 43/43. Next: Stage F (sync-engine architecture audit).
