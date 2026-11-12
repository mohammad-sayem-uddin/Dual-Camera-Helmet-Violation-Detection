# Post-Annotation Stage B — Second Independent Full Audit

- **Stage:** B of the post-annotation mission (Stages A–L), per
  `post_annotation/POST_ANNOTATION_MISSION_BRIEF.md`. Stage A gate: PASSED 15/15.
- **When:** 2026-11-09 (Stage B session), VM mirror `~/workspace/research_helmet`.
- **Object of audit:** the frozen annotation package at
  `repo/data/dual_camera_test_v1/annotations/` (40 files; SHA-256 re-verified at the
  end of this stage: **39/39 OK, package untouched by Stage B**).
- **Method:** (1) full-coverage programmatic audit over all 50 videos / 15,241 frames
  using the package GT + Pass 1 raw data (`work/stageB/step1_programmatic_audit.py`,
  `step1_followup.py`); (2) visual audit from source videos, sequential decode only,
  full-resolution crops (`work/stageB/step2_*`, verdicts in
  `work/stageB/visual_verdicts.csv`). Progress log: `work/stageB/STAGE_B_PROGRESS.md`.
- **Counting bases (Stage A D1 — stated everywhere, never mixed):** *rider track-frame
  basis* = one row per frame a rider (sub)track is active in Pass 1 `track_frames`,
  restricted to GT-retained tracks; *direct head-observation basis* = GT-embedded
  `head_observations`; *GT n_observations basis* = the GT sum (13,062). All three were
  re-derived and reconcile exactly with the package (see §1.1).

## 1. Step 1 — Full-coverage programmatic audit

### 1.1 Basis reconciliation (all exact)
| Basis | Stage B measured | Package/Stage A recorded |
|---|---|---|
| Rider track-frame rows, moto subtracks only | **12,458** | 12,458 ✓ |
| — corrected states on that basis | HELMET 1,825 / NO_HELMET 5,686 / UNCERTAIN 4,947 | identical ✓ |
| GT `n_observations` sum | 13,062 | 13,062 ✓ |
| GT direct head observations | 6,759 | 6,759 ✓ |
| Retained tracks | moto 42F+41B = 83; orphans 71F+54B = 125 | identical ✓ |

### 1.2 Rider subtrack bbox placement — the D3 defect at full coverage
Classification per rider track-frame row (priority: degenerate → off-frame (>100 px
beyond the 3840×2160 frame, the package criterion) → coupled (a GT head observation
exists for that subtrack at that frame) → propagated/stale):

| Basis | Coupled | Propagated/stale | Off-frame | Degenerate | **Non-coupled** |
|---|---|---|---|---|---|
| Moto subtracks (12,458) | 6,154 (49.4%) | 5,726 (46.0%) | 578 (4.6%) | 0 | **6,304 (50.6%)** |
| Moto + orphan (14,249) | 6,752 (47.4%) | 6,834 (48.0%) | 663 (4.7%) | 0 | **7,497 (52.6%)** |

Per-pair non-coupled rate (moto basis) ranges from **16.9% (pair_007)** to **98.1%
(pair_002: 705 stale of 719 rows, only 14 coupled)**; full table in
`work/stageB/placement_by_pair.csv`, per-track detail in
`work/stageB/rider_track_placement.csv`. Mechanism, visible in the per-track data:
many subtracks — concentrated in back-camera views and secondary slots (s1/s2) — were
created by 1–2 head detections and then propagated for their entire lifespan:
**30 rider (sub)tracks have ≤2 coupled observations yet persist ≥30 rows (3,907 rows
in total)**; the extreme case has 1 coupled frame in 273 rows (pair_019
front_moto_0002_rider_s0). Every propagated/stale row had `time_since_update > 0`
(0 exceptions), so the classification is internally consistent. A further 140 rows
overflow the frame edge by ≤100 px (below the off-frame criterion).

### 1.3 Helmet-state consistency (corrected view)
- Moto subtracks: **0 direct HELMET↔NO_HELMET adjacencies** remain — all 73 raw flips
  were inside the 8 SET_CONSTANT subtracks, confirming the Pass 2 correction closure.
- **Orphan tracks were never flip-corrected: 8 direct adjacencies remain**, all in
  pair_004 front, across 5 tracks (front_orphan_0001/0003/0005/0006/0007; frame pairs
  3–4, 11–12, 39–40, 88–89, 100–101, 108–109, 115–116, 118–119). Stage B visual
  checks (§2.3) show the boxes sit on genuinely helmeted standing riders — detector
  state noise, not placement error. Orphans do not feed violation events. → SB-003.

### 1.4 Events, links, references (all clean)
- **Events: 70/70 re-derived exactly** from the recorded rider states under the
  package's stated rule; events CSV ↔ GT JSON states/types match 70/70. No event's
  derivation failed to reproduce.
- **Cross-camera links:** LINKED_BOTH_VIEWS 10/10 have both-side tracks present in
  the GT; **dangling references: 0** across links, events, uncertainty and correction
  files (subtrack IDs included). One bookkeeping inconsistency: the links CSV holds
  70 motorcycle rows (event-derived) vs 68 corrected association results
  (result-derived); the 2 extra UNMATCHED rows (pair_009 front_moto_0001, pair_015
  back_moto_0001) come from GT events whose raw results were dropped as
  FP-referencing. No outcome is wrong; the basis difference is undocumented. → SB-006.
- The 185 off-frame motorcycle boxes in raw rows are exactly the CLIP-flagged set —
  view-corrected as designed; 0 unflagged off-frame moto boxes.

### 1.5 Track-support screen (candidate missed FPs)
Only **2 of 83 retained moto tracks are majority-unsupported** by detections
(time_since_update > 0 on >50% of frames): pair_010 back_moto_0003 (61.9%) and
pair_020 back_moto_0003 (57.1%). Both were visually checked in Step 2 (§2.3):
pair_010's is a **real motorcycle** (cleared); pair_020's is **unresolved** — its box
sits on a distant canopy-shaped (rickshaw-like) object in both checked frames while
the actual motorcycle is a different foreground object. → SB-004 / SB-005.

## 2. Step 2 — Visual audit

### 2.1 Never-inspected motorcycle tracks: **0**
Deriving the inspected set from the actual Pass 2 plan + full-res verdicts and Pass 3
correction-verification + extract-plan records, **all 83 GT-retained motorcycle
tracks were already inspected by direct reference** in Passes 2–3 (verified two
independent ways; the weaker in-span-frame criterion was not needed). There is
therefore no uninspected-track residue for Stage B to clear; the two
majority-unsupported candidates were additionally re-inspected (§1.5).

### 2.2 Fresh stratified sample of rider-bearing frames (coupled stratum)
Design: 5 frames per pair × 25 pairs = **125 frames**, each with a coupled head
observation for the focus subtrack (the judgeable stratum for helmet state), drawn
deterministically (seed 20261109) from frames **not inspected in Passes 2–3** and not
in the targeted set; both cameras represented wherever coupled frames exist.
Results (verdicts: `work/stageB/visual_verdicts.csv` idx 9–133):

| Judgement | Correct | Error | Uncertain |
|---|---|---|---|
| Rider bbox placement | **122** | **3 (2.4%)** | 0 |
| Helmet state (vs corrected GT) | **124** | **0** | 1 |

Placement errors: pair_002 back f287 (box in tree foliage, no head inside — SB-007);
pair_019 back f174 s1 (box on the driver's face instead of the pillion's — SB-008);
pair_019 back f132 s0 (box a narrow sliver on the face edge — SB-009). The single
uncertain state is the foliage frame (no head to judge). **Measured placement-defect
rate in the coupled stratum: 2.4% (3/125); helmet-state error rate: 0/124 decided.**
Sampled frames also independently re-confirm the 8 constant-state corrections
wherever they intersect (pair_002 front, pair_015 front, pair_016 front,
pair_023 front subtracks all visually match their corrected constants).

### 2.3 Targeted checks
- pair_010 back_moto_0003 (f137, f140): box tight on a real motorcycle — **REAL**,
  candidate cleared (SB-005).
- pair_020 back_moto_0003 (f356, f366): box on a distant canopy-shaped object in
  both frames — **UNCERTAIN**, FP suspected, not confirmable at that distance (SB-004).
- pair_004 orphan flip frames (f4/40/89/101/116): placement CORRECT ×5; the heads
  are genuinely helmeted (4/5 clearly; f4 ambiguous) — the orphan flips are detector
  state noise on helmeted standing riders (supports SB-003's recommended fix).

## 3. Comparison with Pass 3's sample-based estimates
Pass 3 (168-frame all-frame sample): frame error 39.3%, of which rider-placement-only
46/66 errors; detection-layer 11.9%. Stage B's full-coverage evidence **confirms the
picture and sharpens its mechanism**:

- The defect is real and structural: **50.6% of all rider track-frame rows
  (moto basis) carry no supporting detection** — the exposure pool from which
  placement errors come is larger than any single sample suggested, because
  propagated boxes persist for entire subtrack lifespans after 1–2 detections.
- But non-coupled ≠ automatically wrong: in the coupled stratum the measured visual
  defect rate is only **2.4%**, and helmet **states** — the quantity the events are
  built from — were wrong in **0 of 124** decided Stage B sample frames and re-derive
  perfectly at track level (70/70 events). Pass 3's 39.3% is a blend over coupled and
  propagated frames; Stage B's decomposition (2.4% coupled-stratum visual defect;
  50.6% of rows structurally unsupported) is consistent with it and explains it.
- Detection layer: Stage B found **no new confirmed FP** among retained tracks
  (1 unconfirmed suspect, SB-004) — consistent with Pass 3's 11.9% detection-layer
  rate being dominated by tracks already excluded by the 69 ops and by raw
  in-frame FP detections (e.g. rickshaws) that the corrected view does not carry
  as tracks.

## 4. Findings (Stage C fix list)
Full records: `work/stageB/second_audit_findings.csv` (11 findings; placement
decision in §5). Tally: **CRITICAL 0 · HIGH 0 · MEDIUM 5 · LOW 5 · UNCERTAIN 1**.

| ID | Sev | Finding |
|---|---|---|
| SB-001 | MEDIUM | 50.6% of rider track-frame rows (moto basis) are non-coupled (stale/off-frame) — rider boxes unusable for spatial evaluation without a validity layer |
| SB-002 | MEDIUM | pair_002 rider placement 98.1% non-coupled (worst pair) |
| SB-003 | MEDIUM | 8 helmet-state flips remain in pair_004 front orphan tracks (never flip-corrected) |
| SB-004 | UNCERTAIN | pair_020 back_moto_0003: suspected FP track (box on distant rickshaw-like object), unconfirmed — Stage C full-frame recheck required before any exclusion |
| SB-005 | LOW | pair_010 back_moto_0003 majority-unsupported but visually REAL — do not exclude on support fraction alone |
| SB-006 | LOW | Links CSV basis inconsistency: 70 event-derived rows vs 68 corrected association results (undocumented) |
| SB-007 | LOW | pair_002 back f287: coupled-frame rider box in tree foliage |
| SB-008 | LOW | pair_019 back f174 s1: rider box on the wrong rider's face |
| SB-009 | LOW | pair_019 back f132 s0: rider box a narrow sliver, head uncovered |
| SB-010 | MEDIUM | 578 rider-subtrack rows (663 incl. orphans) off-frame >100 px were never covered by the moto-only CLIP ops |
| SB-011 | MEDIUM | 30 rider (sub)tracks rest on ≤2 coupled observations across ≥30 rows (3,907 rows) — low-evidence states need an explicit caveat |

**Nothing found in Stage B changes any of the 70 event/violation outcomes** —
hence no CRITICAL findings. The dominant issue remains spatial (bbox placement),
exactly as Pass 3's D3 profile predicted.

## 5. Findings-file placement decision
The brief names `data/dual_camera_test_v1/annotations/audit/second_audit_findings.csv`,
but Stage A §1/§7 define the package as the frozen, checksum-verified 40-file
corrected view whose exact inventory is itself gate-verified. Adding a file would
alter the frozen package. **The findings therefore live only at
`work/stageB/second_audit_findings.csv`; the package was not modified** (re-verified
after Stage B: 39/39 checksums OK, 40 files). Stage E's release freeze can fold the
findings into the release through its own documented process.

## 6. Limitations
1. This is an **AI-assisted audit of an AI-assisted dataset**
   (`AI_ASSISTED_AUDITED`, `human_verified = false`) — not human verification.
2. "Coupled" is defined by the GT's own head-observation coupling; a coupled frame
   can still carry a bad box (measured: 2.4% in the visual sample), and a propagated
   box can still be on-target — the 50.6% figure is a structural exposure measure,
   not a visual error rate.
3. The Step 2 sample is deliberately the coupled stratum (the only stratum where
   helmet state is judgeable); it does not re-estimate Pass 3's all-frame visual
   error rate, and the two rates must not be conflated.
4. Visual verdicts were made from full-resolution frames rendered as crops at
   contact-sheet scale; the pair_020 candidate (SB-004) could not be resolved at
   that scale and is carried as UNCERTAIN, never guessed.
5. Back-camera head-detection sparsity (a Phase 12D property) is the root driver of
   the stale-propagation exposure; Stage B quantifies it but does not re-run or
   alter any approved pipeline module.

**Stage B: COMPLETE.** Next: Stage C (corrections per the fix list above,
provenance preserved, initial release never overwritten).
