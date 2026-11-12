# Post-Annotation Stage C — Correction Report

- **Stage:** C of the post-annotation mission (Stages A–L), per
  `post_annotation/POST_ANNOTATION_MISSION_BRIEF.md`. Inputs: Stage B report
  `docs/phase12/POST_ANNOTATION_SECOND_INDEPENDENT_AUDIT.md` and findings
  `work/stageB/second_audit_findings.csv` (SB-001…SB-011: MEDIUM 5, LOW 5, UNCERTAIN 1;
  zero CRITICAL/HIGH).
- **When:** 2026-11-09/10 (Stage C session), VM mirror `~/workspace/research_helmet`.
- **Hard rule honoured:** the frozen package
  (`repo/data/dual_camera_test_v1/annotations/`, 40 files) and `work/pass1/` were
  **never modified**. Stage C produces a NEW corrections layer in `work/stageC/` for
  Stage E to incorporate into the frozen release. Package integrity re-verified at the
  end of this stage: **checksums 39/39 OK, 40 files**; 0 files under `work/pass1/`
  modified after Stage C start.
- **Counting basis (stated once, used throughout):** rider track-frame basis —
  one row per frame a rider (sub)track is active in Pass 1 `track_frames`, restricted
  to GT-retained tracks; combined moto-subtrack + orphan basis = **14,249 rows**
  (moto-only basis = 12,458).
- **Method for visual adjudications:** sequential decode only (no frame seeking),
  full-resolution (3840×2160) frames extracted once per video
  (`work/stageC/extract_frames.py`, 107 frames), judged from labelled full-res crops
  (`work/stageC/crops/`). SB-003 was adjudicated using **genuinely coupled frames only**
  (frames with a GT head observation for that track).

## 1. Deliverables of this stage

| File | Content |
|---|---|
| `work/stageC/rider_frame_validity.csv` | 14,249-row rider-frame validity layer (SB-001/002/010/011 + SB-007/008/009 overrides) |
| `work/stageC/corrections_stagec.csv` | 17-row Stage C corrections log (Pass 2 column conventions + `finding_id`); every finding dispositioned |
| `work/stageC/sb003_coupled_frame_verdicts.csv` | 53 coupled track-frame visual verdicts (49 unique frames), pair_004 front orphans |
| `work/stageC/sb004_span_verdicts.csv` | 35 whole-span visual verdicts, pair_020 back_moto_0003 |
| `work/stageC/sb007_009_neighbor_spotcheck.csv` | 23 neighbour-frame spot-check outcomes (SB-007/008/009) |
| `work/stageC/stagec_verification.json` | 11/11 consistency checks PASS (`verify_stagec.py`) |
| `work/stageC/validity_layer_summary.json`, `corrections_summary.json` | Count summaries |
| `work/stageC/STAGE_C_PROGRESS.md` | Checkpointed progress log |

## 2. The rider-frame validity layer (SB-001, SB-010, SB-011, SB-002)

Built with the Stage B Step-1 classification algorithm unchanged (priority:
degenerate → off-frame >100 px → coupled → propagated/stale), so its base counts
reproduce Stage B **exactly** before Stage C overrides:

| Class | Stage B base | Stage C final | spatial_use_valid |
|---|---|---|---|
| COUPLED | 6,752 | 6,748 | true, except pair_002 (SB-002) |
| PROPAGATED_STALE | 6,834 | 6,834 | false |
| OFF_FRAME | 663 | 663 | false |
| INVALID (Stage C override) | — | 4 | false |
| **Total** | **14,249** | **14,249** | **true: 6,727 / false: 7,522** |

Columns: `pair,cam,track,subtrack,frame,placement_class,spatial_use_valid,
evidence_class,finding_ref,stage_c_note`. For orphan tracks, `track` and `subtrack`
both carry the orphan track ID (orphans have no subtrack split).

- **SB-001 (validity layer):** `spatial_use_valid` is true **only** for coupled rows
  (6,727 rows after the pair_002 flag and the 4 invalidations). The layer alters no
  box or state value; it marks non-coupled rows invalid for **spatial** use. Stage K
  must not compute rider-box IoU over rows with `spatial_use_valid=false`.
- **SB-010 (off-frame rider rows):** all 663 off-frame rows (578 moto-subtrack +
  85 orphan) carry `placement_class=OFF_FRAME` and `finding_ref=SB-010` — the
  rider-level coverage the Pass 2 moto-only CLIP ops never provided.
- **SB-011 (low-evidence subtracks):** `evidence_class=LOW_EVIDENCE_SUBTRACK` on
  exactly **30 (sub)tracks / 3,907 rows** (definition reproduced from Stage B
  aggregates: ≤2 coupled observations across ≥30 track-frame rows). Their track-level
  states rest on 1–2 direct observations plus propagation and must carry an explicit
  low-observation caveat in the Stage E release.
- **SB-002 (pair_002):** every pair_002 rider row in the layer (760 combined-basis
  rows; 719 on the moto basis, 98.1% non-coupled there) has
  `spatial_use_valid=false`, including its coupled rows — the pair's one
  coupled-stratum sample frame was itself defective (SB-007). **States are explicitly
  retained, not altered:** they were visually verified at coupled frames in Stage B
  (helmet-state error 0/124 decided sample frames).

## 3. Per-finding dispositions

### SB-001 — MEDIUM — rider bbox placement exposure → CORRECTED (layer)
Validity layer built and verified (§2). No box or state re-estimated anywhere.

### SB-002 — MEDIUM — pair_002 placement → CORRECTED (pair flag, states retained)
See §2. Recorded in the corrections log as `FLAG_PAIR_SPATIALLY_UNUSABLE` (SC-002).

### SB-003 — MEDIUM — pair_004 front orphan helmet flips → CORRECTED IN PART; 3 flips carried UNCERTAIN
Adjudicated per track at full resolution using **only genuinely coupled frames**
(53 track-frame verdicts over 49 unique coupled frames;
`sb003_coupled_frame_verdicts.csv`). Result — three tracks are constant, two are not:

| Track | Coupled frames judged | Visual truth | Op |
|---|---|---|---|
| front_orphan_0005 | 4 (f80, 81, 89, 91) | HELMET 4/4 (worn black helmet, rear view) | `SET_CONSTANT_HELMET_STATE` → HELMET |
| front_orphan_0006 | 16 (f81–f111) | HELMET 16/16 | `SET_CONSTANT_HELMET_STATE` → HELMET |
| front_orphan_0007 | 14 (f111–f128) | HELMET 14/14 | `SET_CONSTANT_HELMET_STATE` → HELMET |
| front_orphan_0001 | 9 (f0–f15) | held beside bare head f0–f4 → donning f11–f12 → worn f13–f15 | `MARK_OBSERVATIONS_UNCERTAIN` (f3–f4, f11–f12) |
| front_orphan_0003 | 10 (f31–f42) | held beside bare head f31–f37 → donning f39 → worn f40–f42 | `MARK_OBSERVATIONS_UNCERTAIN` (f39–f40) |

The five tracks are fragments of one standing rider filmed putting on a helmet and
then sitting on the motorcycle. For 0005/0006/0007 the detector's `no_helmet` calls
are pure noise on a constantly helmeted rider. For 0001/0003 the flips sit on top of a
**genuine donning transition**: no constant state can be honestly assigned, so —
exactly as the finding's fallback prescribes — the flipped observations are marked
UNCERTAIN and carried to Stage D; the packaged track states are left unchanged and
are **not** endorsed for those two tracks. Flip accounting after the ops:
**5 of 8 adjacencies resolved by constant ops, 3 explicitly carried as UNCERTAIN,
0 unaccounted.**

*Refinement of Stage B, recorded honestly:* Stage B's targeted check (one frame per
track: f4/40/89/101/116) concluded the orphan flips were detector noise on helmeted
riders. The coupled-frame-complete Stage C check confirms that for the three seated
tracks and for the sampled frames, but shows 0001/0003 span the actual donning —
the held-helmet frames (e.g. f4, which Stage B sampled as "ambiguous") are genuinely
unhelmeted moments. Both statements are preserved here and in the evidence CSV.

### SB-004 — UNCERTAIN → CONFIRMED FP (pair_020 back_moto_0003)
Full-frame recheck of the **whole span (f354–f388, 35/35 frames)** at full resolution
(`sb004_span_verdicts.csv`): in every frame the track box sits on **a man riding a
bicycle loaded with orange cargo sacks** — thin bicycle wheels and pedals are visible,
there is no engine or motorcycle body; at f386–f388 the bicycle is progressively
occluded by the foreground motorcycle rider and the box remains on the sack load.
(At Stage B's crop scale the sack load read as a distant canopy-shaped object; the
object is in fact a cargo bicycle.) Op: `EXCLUDE_FP_TRACK` (SC-006), provenance
`STAGE_C_CORRECTION`. **Consequence flagged by the finding itself:** the frozen
package still contains this track; when Stage E incorporates the exclusion,
pair_020's event inventory changes and must be re-derived there. Stage D should
confirm no other retained track is the same bicycle.

### SB-005 — LOW — pair_010 back_moto_0003 → ADJUDICATED NO-OP (by design)
No action: majority-unsupported (61.9%) but visually REAL (Stage B, box tight on a
real motorcycle). Recorded in the log as `ADJUDICATED_NO_OP` (SC-007) with the rule
spelled out — support fraction alone never excludes a track; exclusion requires
visual FP confirmation (contrast SB-004) — so Stage D does not re-litigate it.

### SB-006 — LOW — links counting basis → DOCUMENTARY (no data change)
No data changed. The correction is the following exact reconciliation text, recorded
in the corrections log (SC-008) and reproduced here verbatim for Stage E's release
notes:

> COUNTING BASIS (for Stage E release notes): cross_camera_identity_links.csv is event-derived - it carries one motorcycle row per global motorcycle event in dual_camera_gt_events.csv (70 rows). The corrected Phase 12E association recount is result-derived - it counts the association results that survive the corrected view (68 motorcycle results) after dropping results that reference FP-excluded tracks. The 2-row difference is exactly the event-derived UNMATCHED rows for pair_009 front_moto_0001 and pair_015 back_moto_0001, whose raw association results were dropped as FP-referencing. No link status is affected: the 10 LINKED_BOTH_VIEWS rows have both-side tracks present in the GT and dangling references are 0. The two bases must not be mixed: use 70 when counting events/links as packaged, 68 when counting corrected association outcomes.

### SB-007 — LOW — pair_002 back f287 rider box in foliage → INVALIDATED + spot-checked
Validity layer: frame marked `placement_class=INVALID`, `spatial_use_valid=false`.
Neighbour spot-check (f283–f291, `sb007_009_neighbor_spotcheck.csv`): the subtrack
does not exist before f287 (absent f283–f286); the propagated box then **remains in
the tree canopy with no head inside at f288, f289, f290 and f291** — the defect is
broader than a single frame at the subtrack's birth, and those neighbours are already
spatial-invalid as PROPAGATED_STALE under SB-001/SB-002.

### SB-008 — LOW — pair_019 back f174 s1 box on driver's face → INVALIDATED + spot-checked
Frame marked INVALID. Neighbour spot-check (f171–f177): the s1 box is squarely on
the pillion's head in **6/6** neighbouring frames — f174 is an isolated single-frame
placement error. Helmet state unaffected (both riders bare-headed).

### SB-009 — LOW — pair_019 back f132 s0 sliver box → INVALIDATED + spot-checked (window defective)
Frame marked INVALID. Neighbour spot-check (f129–f135) shows the s0 box is the
**same narrow sliver across the whole window**: f129 (coupled, tsu=0) covers only the
rear edge of the driver's head and is therefore **also marked INVALID** (4th override,
SC-011); f130/f131/f133 slivers are stale and already spatial-invalid; **f134 and
f135 contain no head at all** (sliver floating on background), likewise already
spatial-invalid as PROPAGATED_STALE. State NO_HELMET unaffected (underlying head
detection at f132 is correct).

### SB-010 — MEDIUM — off-frame rider rows → CORRECTED (layer flags)
See §2: 663 rows flagged OFF_FRAME with SB-010 refs.

### SB-011 — MEDIUM — low-evidence subtracks → CORRECTED (layer flags)
See §2: 30 (sub)tracks / 3,907 rows flagged LOW_EVIDENCE_SUBTRACK; release caveat
required at Stage E.

## 4. Corrections log summary

`work/stageC/corrections_stagec.csv` — **17 rows**, provenance `STAGE_C_CORRECTION`
throughout: ADD_RIDER_VALIDITY_LAYER 1 · FLAG_PAIR_SPATIALLY_UNUSABLE 1 ·
SET_CONSTANT_HELMET_STATE 3 · MARK_OBSERVATIONS_UNCERTAIN 2 · EXCLUDE_FP_TRACK 1 ·
ADJUDICATED_NO_OP 1 · DOCUMENT_BASIS_NOTE 1 · INVALIDATE_RIDER_BOX_FRAME 4 ·
FLAG_OFF_FRAME_RIDER_ROWS 1 · FLAG_LOW_EVIDENCE_SUBTRACKS 1 · SPOT_CHECK_RECORD 1.
All 11 findings (SB-001…SB-011) are dispositioned; none is dispositioned twice with
conflicting outcomes.

## 5. What was NOT changed, and why

- **Frozen package: untouched.** 40 files, checksums re-verified **39/39 OK** after
  all Stage C work. Every Stage C output lives in `work/stageC/`; incorporation is
  Stage E's documented process, not Stage C's.
- **Pass 1 (`work/pass1/`): untouched** (0 files modified after Stage C start).
  Original videos and Phase 11 metadata: read only.
- **No bounding box was re-estimated, moved, or redrawn anywhere.** Stage C marks
  rows/frames invalid for spatial use and corrects helmet-state constants only where
  coupled-frame visual evidence is unanimous; defective boxes stay as recorded, with
  validity flags carrying the correction.
- **No helmet state was changed on propagated evidence.** SB-003 constants rest only
  on genuinely coupled frames; the two donning-transition tracks got no constant.
- **pair_010 back_moto_0003 was not excluded** (SB-005 no-op by design), and
  **pair_020's packaged event inventory was not altered** — it changes only when
  Stage E incorporates the SB-004 exclusion.

## 6. Verification (all PASS — `work/stageC/stagec_verification.json`, 11/11)

1. Validity rows 14,249 == Stage B placement rows (14,249) == step1 moto+orphan total.
2. Class reconciliation: base 6,752/6,834/663 reproduced; final 6,748/6,834/663 + 4 INVALID.
3. `spatial_use_valid=true` occurs only on COUPLED rows (6,727 rows).
4. pair_002: all 760 rows `spatial_use_valid=false`.
5. LOW_EVIDENCE_SUBTRACK: exactly 30 (sub)tracks / 3,907 rows.
6. INVALID frames are exactly the 4 intended (SB-007 f287; SB-008 f174; SB-009 f132, f129).
7. All 663 OFF_FRAME rows carry the SB-010 ref.
8. All 11 findings dispositioned in the corrections log (17 rows).
9. SB-003 ops: SET_CONSTANT on 0005/0006/0007; MARK_UNCERTAIN on 0001/0003.
10. SB-004 EXCLUDE_FP_TRACK op present for pair_020 back_moto_0003.
11. SB-003 flip accounting: 5 resolved + 3 carried UNCERTAIN = 8 total, 0 unaccounted.

## 7. Unresolved carry-forwards for Stage D

1. **SB-003 residual (UNCERTAIN by design):** front_orphan_0001 (flips at f3–f4,
   f11–f12) and front_orphan_0003 (flip at f39–f40) span a genuine helmet-donning
   transition. Their flipped observations are marked UNCERTAIN in the Stage C layer;
   their packaged track-level states are unendorsed. Orphans do not feed violation
   events, so no event outcome is affected.
2. **SB-004 downstream:** the pair_020 back_moto_0003 exclusion exists only in the
   Stage C layer until Stage E incorporation; Stage E must re-derive pair_020's
   event inventory then, and Stage D should confirm the same cargo bicycle is not
   carried by any other retained track.
3. **Layer incorporation (Stage E):** the validity layer, the 3 constant-state ops,
   the 2 uncertain-markings, the 4 frame invalidations and the SB-006 basis text
   must be folded into the `dual_camera_gt_v1.0_ai_assisted_audited` release with
   this report as provenance; Stage K must filter rider-spatial metrics by
   `spatial_use_valid` and carry the SB-011 low-observation caveat.

## 8. Limitations

1. This is an **AI-assisted correction of an AI-assisted dataset**
   (`AI_ASSISTED_AUDITED`, `human_verified = false`) — not human verification.
2. SB-003 adjudication used coupled frames only, by design; states between coupled
   frames of the constant tracks are inferred from the track's unanimous coupled
   evidence, the same policy Pass 2 applied to moto subtracks.
3. During the donning frames themselves (0001 f11–f12, 0003 f39) the rider's state
   is genuinely intermediate; UNCERTAIN is the honest label, not a measurement gap
   that more resolution would close.
4. Neighbour spot-checks were bounded to the mandated windows (±3–4 frames around
   SB-008/SB-009 frames; subtrack birth window for SB-007); other frames of those
   subtracks are covered by the validity layer's class flags, not by individual
   visual verdicts.

**Stage C: COMPLETE.** All 11 Stage B findings dispositioned (10 corrected/recorded
in the Stage C layer, 1 documentary, 1 no-op by design, with the SB-003 donning
residue explicitly carried as UNCERTAIN). Next: Stage D (third/final full audit).
