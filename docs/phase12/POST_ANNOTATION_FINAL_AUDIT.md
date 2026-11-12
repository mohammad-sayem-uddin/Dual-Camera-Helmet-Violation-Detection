# Post-Annotation Mission — Stage D: Third and Final Full Audit

**Date:** 2026-11-09 · **Stage:** D of A–L · **Inputs (all READ-ONLY, unmodified):** frozen package
`data/dual_camera_test_v1/annotations/` (40 files), `work/pass2/corrections.csv` (63 ops) +
`work/pass3/corrections_addendum.csv` (6 ops), Stage C layer `work/stageC/`
(`corrections_stagec.csv`, 17 rows; `rider_frame_validity.csv`, 14,249 rows), Stage B audit report,
Stage C correction report. **Evidence:** `work/stageD/` (scripts, JSON results, verdict CSVs,
extracted frames, judge sheets). Package checksums verified **39/39 OK at stage start and at
stage end**; the package, both Mission-1 corrections files, `work/pass1/`, the original videos
and all Phase 11 metadata are byte-unchanged.

> Dataset status remains **`AI_ASSISTED_AUDITED`** — AI-generated with AI audit passes. Nothing in
> this stage (or any stage) makes it human-verified ground truth, and the release must say so.

---

## 1. Validity-layer reproduction — PASS (exact)

The rider placement classes were recomputed **independently** (Stage D's own script,
`work/stageD/step1_validity_repro.py`; documented rule — degenerate → off-frame (>100 px beyond
the 3840×2160 raster) → coupled (a genuine GT head observation exists at that frame) →
propagated_stale) for a stratified subset of **7 pairs** (001, 002, 004, 007, 013, 019, 020 —
including both audit-critical pairs 002 and 019):

- **Base class counts match Stage B's `placement_by_pair.csv` EXACTLY for every pair and class**
  (e.g. pair_002 COUPLED 22 / STALE 738 / 760; pair_019 500 / 665 / OFF 97 / 1,262;
  pair_020 138 / 194 / 14 / 346).
- Row-level diff of the recomputation against the Stage C final validity layer on those pairs:
  **exactly the 4 documented INVALID overrides** (pair_002 back_rider_s0 f287; pair_019
  back_rider_s0 f129, f132; back_rider_s1 f174) and **zero other differences; zero missing rows**.
- Global totals confirmed from the Stage C CSV itself: 14,249 rows; **COUPLED 6,748 /
  PROPAGATED_STALE 6,834 / OFF_FRAME 663 / INVALID 4**; **spatial_use_valid = true on exactly
  6,727 rows** — all of them COUPLED, **none in pair_002** (the SB-002 exclusion is correctly
  applied); LOW_EVIDENCE_SUBTRACK on exactly 30 (sub)tracks / 3,907 rows.

The validity layer is arithmetically sound and faithfully encodes Stage B's findings.

## 2. SB-003 operations verification — ALL 5 VERIFIED (own eyes, own extraction)

All coupled frames of the five pair_004 front orphan tracks were re-extracted at full resolution
by sequential decode and judged independently of Stage C's verdicts (53 track-frame units;
`work/stageD/sb003_verification.csv`):

| Track | Stage C op | Stage D verdict |
|---|---|---|
| front_orphan_0005 | SET_CONSTANT_HELMET_STATE | **CONFIRMED** — helmet worn in 4/4 coupled frames |
| front_orphan_0006 | SET_CONSTANT_HELMET_STATE | **CONFIRMED** — helmet worn in 16/16 coupled frames |
| front_orphan_0007 | SET_CONSTANT_HELMET_STATE | **CONFIRMED** — helmet worn in 14/14 coupled frames |
| front_orphan_0001 | MARK_OBSERVATIONS_UNCERTAIN | **CONFIRMED** — genuine donning transition: helmet held in hand beside the bare head f0–f4 → being put on f11–f12 → worn f13–f15. **No constant state is honestly assignable.** |
| front_orphan_0003 | MARK_OBSERVATIONS_UNCERTAIN | **CONFIRMED** — continuation of the same action: held/being positioned f31–f37 → worn f39–f42. **No constant state is honestly assignable.** |

## 3. SB-004 downstream check — CONFIRMED CLEAN

Mid-frames of **29 retained tracks** were inspected (`work/stageD/sb004_downstream_scan.csv`):

- **pair_020 (both cameras):** no other retained track is the cargo bicycle. Independently
  re-confirmed from Stage D's own extraction that the excluded back_moto_0003's mid-frame (f371)
  sits exactly on the man riding a bicycle loaded with orange/red sacks. In back_moto_0002's
  mid-frame that same bicycle passes **unboxed in the background** — no retained track carries it.
- **pairs 019 and 021 (back):** retained tracks are real motorcycles and pedestrian/rider heads
  only; **the cargo bicycle appears nowhere**.

## 4. Event / link re-derivation with ALL layers — the pair_020 delta, exactly

The package builder's event/link logic (`build_package.py` Part 2) was replicated read-only
against the frozen inputs (`work/stageD/step4_events.py`).

**View A (69 ops only) reproduces the package EXACTLY:** 70 events with **0 field mismatches**
(type, status, outcome, driver/pillion states, and the deterministic global event IDs all match),
links CSV **202 rows, multiset-equal**, recount LINKED 10 / UNMATCHED 34 / INSUFFICIENT 19 /
ABSTAIN 5, dropped moto 17 / orphan 5. The frozen package is exactly what its inputs produce.

**View B (69 ops + Stage C layer) — delta is exactly ONE event, fully explained:**

| | Packaged (View A) | All-layers (View B) |
|---|---|---|
| Event | `pair_020_moto_01`: tracks front_moto_0001 + back_moto_0003, status AMBIGUOUS_ABSTAIN, link AMBIGUOUS_HUMAN_UNCERTAIN, type DRIVER_NO_HELMET | `pair_020_moto_01`: track front_moto_0001 alone, status UNMATCHED, link FRONT_VIEW_ONLY, type DRIVER_NO_HELMET |
| Event total | 70 | **70 (unchanged)** |
| Type counts | DRIVER 27 / BOTH 10 / PILLION 3 / NO_VIOLATION 6 / INSUFFICIENT 24 | **identical** |
| Link totals | LINKED 10 / FRONT_ONLY 28 / BACK_ONLY 27 / AMBIGUOUS 5 | LINKED 10 / FRONT_ONLY **29** / BACK_ONLY 27 / AMBIGUOUS **4** |
| Links rows | 202 | **202** (1 row swapped) |
| Recount | ABSTAIN 5, dropped moto 17 | ABSTAIN **4**, dropped moto **18** |
| Event IDs | — | **no renumbering anywhere** (the replacement event takes the same ID `pair_020_moto_01` by sort order) |

**Why the violation outcome survives:** back_moto_0003 has **no rider subtracks at all**, and its
packaged association result carried `rider_associations = []` — it contributed nothing to the
event's helmet evidence. The DRIVER_NO_HELMET outcome came entirely from front_moto_0001's
slot-0 subtrack (131 observations). Excluding the bicycle therefore changes the event's *link
status* (ambiguous → front-only) but **not its violation type** — precisely the change Stage C
predicted, and the only change in the whole inventory.

## 5. Fresh defect sweep (never-inspected frames) + programmatic re-run

**Design:** 125 coupled rider frames (5 per pair, stratified by recorded class, deterministic
seed 20261109), verified disjoint from every prior inspection — the prior-inspected union
(Pass 2 review plan + full-res verdicts, Pass 3 sample + correction verifications, Stage B
verdicts + worklist, Stage C SB-003/004/007–009 verdicts) is **640 distinct (pair, cam, frame)**
keys. Frames extracted by sequential decode; judged on labelled crops (white = rider box,
magenta = GT head-observation box). Verdicts: `work/stageD/fresh_sample_verdicts.csv`.

**Measured results (n = 125):**

| Axis | CORRECT | ERROR | UNCERTAIN | Error rate |
|---|---|---|---|---|
| Rider placement (coupled stratum) | 123 | 2 | 0 | **1.6%** |
| Helmet state | 117 | 4 | 4 | **3.2%** of all; **3.3%** of decided (121) |

- Placement errors (both the known background-object family): pair_006 back f50
  (back_orphan_0002 — head box in treetop foliage) and pair_023 front f6
  (front_moto_0001_rider_s0 — head box on a background signboard).
- State errors — **all four at long-range / track-edge frames**: pair_011 front f1 and
  pair_016 front f231 and pair_015 front f0 (orphan) recorded NO_HELMET where a helmet is
  visibly worn (in each case the same track/rider is HELMET in its other sampled coupled
  frames); pair_023 front f6 recorded HELMET spuriously (the signboard box above; the rider is
  bare-headed). Uncertain ×4: distance (pair_002 f3, pair_012 f1), the pair_004 donning
  transition (f23), no-head-in-box (pair_006 f50).
- Comparison: Stage B's fresh coupled sample measured placement error 2.4% and 0/124 state
  errors. Stage D's placement rate is consistent (same order); the 4 state errors are a real,
  newly-measured residual concentrated in faint edge frames. No single one can change an
  event: event derivation requires ≥3 coupled observations of one class per rider.

**Programmatic full-coverage re-run** (`work/stageD/step5_programmatic.json`):

- Direct HELMET↔NO_HELMET adjacencies: **L1 (69 ops) = 8** — exactly Stage B's pair_004 orphan
  set; **L2 (+ Stage C constant ops) = 3**; **L3 (+ Stage C MARK_UNCERTAIN windows) = 0 literal
  adjacencies**. The only residue is the two adjudicated donning-transition windows.
- Dangling references (links + events, resolved against GT tracks ∪ subtracks ∪ all exclusion
  layers): **0**. References pending Stage E incorporation: **exactly 2** — the packaged links
  row and events row naming pair_020 back_moto_0003, which the §4 delta removes.
- Spatially-valid subset (6,727 rows): **0 off-frame boxes** under the >100 px criterion;
  61 rows carry minor edge overflow ≤ 100 px (disclosure figure).

## 6. Freeze-readiness checklist for Stage E — release `dual_camera_gt_v1.0_ai_assisted_audited`

The release is buildable **from verified layers only**. It MUST incorporate:

- [ ] The frozen package as the base (byte-identical starting point; checksums 39/39).
- [ ] All **69 Mission-1 correction ops** (63 Pass 2 + 6 Pass 3 addendum) — already embodied in
      the package; keep their provenance records.
- [ ] All **17 Stage C rows** (`corrections_stagec.csv`), in particular **SC-008**
      (EXCLUDE pair_020 back_moto_0003) — applying it produces **exactly the §4 delta**
      (one event re-derived as front-only, same ID, same type; totals 70 events,
      type counts unchanged; links 202 with FRONT_ONLY 29 / AMBIGUOUS 4).
- [ ] The **rider-frame validity layer** (14,249 rows; Stage D-verified): COUPLED 6,748 /
      PROPAGATED_STALE 6,834 / OFF_FRAME 663 / INVALID 4, `spatial_use_valid=true` on exactly
      6,727 rows, pair_002 spatially unusable, 30 low-evidence (sub)tracks flagged.
- [ ] The **SB-006 reconciliation text verbatim** (Stage C report §3 / SC-010): links CSV
      202 rows = 70 event-derived moto rows + 132 rider rows vs 68 corrected cross-camera
      association results; the 2 extra UNMATCHED rows (pair_009 front_moto_0001,
      pair_015 back_moto_0001) come from the one-result-per-retained-track recount.
- [ ] Rebuilt events/links artifacts from the all-layers view (View B), with event IDs
      preserved (no renumbering) and a release note recording the single pair_020 change.
- [ ] Plate fields as NOT_ASSESSED (0 plates recorded; plate visibility only, never plate text).
- [ ] The dataset label **`AI_ASSISTED_AUDITED`** and the Phase 12B protocol-tension disclosure
      (carried since Mission 1).

## 7. Unresolved items the release must DISCLOSE (not fix)

1. **Propagated-box defect profile (intrinsic, flagged not fixed):** ~half of rider track-frame
   rows are non-coupled (Stage B: 50.6% on the moto basis; validity layer: 6,834 STALE + 663
   OFF_FRAME + 4 INVALID of 14,249). Spatial/statistical use is licensed only on the
   6,727 `spatial_use_valid` rows. No box was ever re-estimated in any stage.
2. **pair_002:** all 760 combined rows spatially unusable (states retained); event stays
   INSUFFICIENT_EVIDENCE.
3. **pair_004 donning transitions:** states for front_orphan_0001/0003 inside the transition
   windows are UNCERTAIN by adjudication (Stage C + Stage D concur); the packaged per-frame
   values inside the windows are explicitly **not endorsed**.
4. **Low-evidence subtracks:** 30 (sub)tracks rest on ≤2 coupled observations over ≥30 rows.
5. **Residual per-frame state error (Stage D measurement):** 4 wrong helmet-state values remain
   in the frozen GT head observations (pair_011 front f1, pair_015 front f0 orphan, pair_016
   front f231 — NO_HELMET recorded, helmet worn; pair_023 front f6 — HELMET recorded
   spuriously), plus 2 residual background-object head boxes (pair_006 back f50 foliage,
   pair_023 front f6 signboard). Measured fresh-sample rates: state 3.3% of decided,
   placement 1.6% (coupled stratum). These are disclosed residuals of the frozen package;
   the correction-op vocabulary of Stages C–D does not include rewriting individual GT head
   observations, and no event outcome depends on any single one.
6. **pair_020 front_orphan_0002** mid-frame box mostly off-frame on background (Step 3
   observation; orphan tracks do not feed events).
7. **61 spatially-valid rows** carry minor (≤100 px) frame-edge box overflow.
8. **Counting bases differ by artifact** (Stage A D1): 12,458 moto track-frame / 14,249 combined
   / 13,062 GT n_observations / 6,759 GT embedded direct head observations — never mix bases.
9. **Mac sync and git commit/push** of the whole post-annotation work remain pending (outside
   this stage's gate).

## 8. Verdict

**READY for the Stage E freeze.** Every Stage C claim Stage D was asked to test reproduced
exactly or was confirmed by independent visual inspection: the validity layer is arithmetically
exact; all five SB-003 dispositions are correct (three constants confirmed frame-by-frame; two
genuine donning transitions correctly left UNCERTAIN); SB-004's exclusion is downstream-clean
and independently re-confirmed; the package is exactly reproducible from its inputs, and the
all-layers event inventory differs by precisely the one predicted, fully-explained pair_020
change with **no change to event count, type counts, or event IDs**. The fresh-sample residuals
(§7.5) are small, measured, concentrated in known weak strata, and belong in the release's
disclosure — they do not block a freeze whose whole design is *frozen base + layered,
provenance-preserving corrections + explicit validity flags*.

**Limitations of this audit:** the fresh visual sample is 125 frames of the 6,748 coupled
rider-frames (all prior stages' samples were disjoint and consistent); back-camera and
rear-view evidence remains intrinsically weaker (absence of a detection is never treated as
absence of a rider); Stage D judged helmet state only where a genuine head observation couples
to the frame.
