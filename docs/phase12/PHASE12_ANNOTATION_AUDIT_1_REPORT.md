# Phase 12 — Annotation Audit 1 Report (Pass 2)

**Dataset:** dual_camera_test_v1 — 25 synchronized front/back pairs, 50 videos, 15,241 decoded frames
**Pass 1 release:** `work/pass1/` (immutable; not modified by this audit)
**Audit type:** AI-assisted audit. The dataset remains labelled **AI_ASSISTED_AUDITED** — it is **not** human-verified ground truth.
**Audit layers:** (1) programmatic checks over 100% of Pass 1 outputs, (2) visual inspection of 258 targeted contact-sheet cells, (3) full-resolution re-inspection of all 88 cells the sheet pass could not resolve.
**Artifacts:** `work/pass2/programmatic_findings.csv`, `work/pass2/visual_verdicts.csv`, `work/pass2/fullres_verdicts.csv`, `work/pass2/fullres_crops/` (141 manifest crops), `work/pass2/corrections.csv`, `work/pass2/corrections_summary.json`.

All numbers below are measured from the artifacts above.

---

## 1. Programmatic audit (Pass 1 outputs, 100% of frames)

220 findings, all in three check families. **Zero** findings in the structural families (coverage counts, frame-index contiguity, PTS monotonicity, detection bbox/confidence validity, track referential integrity, association-vs-summary counts — all reconcile).

| Check | Findings | Severity | Meaning |
|---|---|---|---|
| C4_TRACK_BBOX_OFFFRAME | 185 frame-rows over **20 distinct tracks** | LOW | Track bbox extends beyond frame bounds (Kalman tail after the target exits) |
| C5_HELMET_STATE_FLIP | 8 rider subtracks, **73 direct HELMET↔NO_HELMET flips** | MEDIUM | Per-frame helmet class alternates within one rider subtrack |
| C6_FRAGMENT_TRACK | 27 moto tracks with hits = 1 | LOW | Single-observation tracks: real brief motorcycles or false positives |

Reference totals from the same pass: detections — motorcycle 12,156 / person 22,350 / head 9,805 over 15,241 frames; moto tracks 100; orphan (rider) tracks 130; rider helmet observations — NO_HELMET 5,692 / UNCERTAIN 5,385 / HELMET 1,527.

## 2. Contact-sheet visual inspection (258 targeted cells)

Cells were sampled where the programmatic audit or track structure demanded a human-style look: every flip frame (FLIP), every fragment track (FRAG), the mid-frame of every moto track (FRONT/BACK), and every third orphan track (ORPHAN).

| Tag | Cells | CORRECT | ERROR | UNCERTAIN |
|---|---|---|---|---|
| FRONT | 48 | 46 | 0 | 2 |
| BACK | 52 | 43 | 4 | 5 |
| ORPHAN | 58 | 53 | 2 | 3 |
| FRAG | 27 | 19 | 3 | 5 |
| FLIP | 73 | 0 | 0 | 73 |
| **Total** | **258** | **161** | **9** | **88** |

All 9 sheet-level ERRORs are FP_DETECTION: pair_009 back systematically boxes auto-rickshaws as motorcycles (back_moto_0001/0002/0003), pair_012 back_moto_0001 sits on a rickshaw's front wheel assembly, and two orphan tracks (pair_002 back_orphan_0001, pair_023 back_orphan_0001) contain no person at all.

All 73 FLIP cells were UNCERTAIN at sheet scale — the riders' heads are too small in a 620 px contact-sheet cell to judge helmet state. The remaining 15 UNCERTAIN cells (FRONT 2, BACK 5, ORPHAN 3, FRAG 5) were likewise resolution- or ambiguity-limited.

## 3. Full-resolution resolution of the 88 uncertain cells

Each uncertain frame was re-extracted from the source video at native 3840×2160 (sequential decode only, no frame seeking), cropped around the referenced track's box (×2.5 expansion; ×2.0/×1.4 for context crops), and re-inspected. For FLIP subtracks, context crops at the largest **genuinely observed** head boxes (frames where a head detection actually couples to the subtrack, not tracker-propagated boxes) established each rider's true state.

**Result: 88 of 88 resolved. 0 remain uncertain.**

| Group | Cells | Outcome |
|---|---|---|
| FLIP | 73 | **FLIP_NOISE 73 / FLIP_REAL 0 / STILL_UNCERTAIN 0** |
| Non-FLIP (FRONT/BACK/ORPHAN/FRAG) | 15 | **ERROR 15 (all FP_DETECTION) / CORRECT 0 / STILL_UNCERTAIN 0** |

### 3.1 FLIP outcomes per rider subtrack

Every one of the 73 audited flips is detector noise: in every flip-frame crop the rider's head shows the same state as in the large-frame context crops. No rider changes helmet state mid-clip in any of the 8 subtracks.

| Subtrack | Flips | Constant true state | Decisive evidence |
|---|---|---|---|
| pair_002 front front_moto_0001_rider_s0 | 2 | NO_HELMET | Bare hair + sunglasses visible in f0/f2/f4 crops (head only ~50 px — weakest-evidence group, see §6) |
| pair_010 back back_moto_0001_rider_s0 | 8 | HELMET | Helmet shell/visor visible in all 8 flip crops; f172 context (83×108 px head) |
| pair_011 front front_moto_0001_rider_s0 | 7 | HELMET | Helmet visible in all 7 flip crops; f196 context (322×344 px head) |
| pair_012 front front_moto_0004_rider_s0 | 3 | HELMET | Helmet visible in all 3 flip crops; f237 context (248×260 px head) |
| pair_015 front front_moto_0002_rider_s0 | 11 | HELMET | Helmet visible in all 11 flip crops; f285 context (267×281 px head) |
| pair_016 front front_moto_0001_rider_s0 | 36 | HELMET | Helmet visible in all 36 flip crops; f475 context (400×454 px head) unambiguous |
| pair_023 front front_moto_0001_rider_s0 | 4 | NO_HELMET | Bare curly hair in all 4 flip crops; f232 context (339×379 px head). A spare helmet hangs on the handlebar — carried, not worn |
| pair_024 back back_moto_0001_rider_s0 | 2 | NO_HELMET | Bare hair (rear view) in both flip crops; f566 context (253×245 px head) |

### 3.2 Non-FLIP outcomes (15 cells, all false positives)

- **Auto-rickshaw boxed as motorcycle:** pair_009 back back_moto_0009 (f390), back_moto_0010 (f392), back_moto_0011 (f404) — 6 cells (FRAG+BACK pairs). The referenced boxes cover the colourful three-wheeler at frame right, **not** the real distant motorcycle at centre (which is separately and correctly tracked as back_moto_0004).
- **Squatting pedestrian boxed as motorcycle:** pair_009 back back_moto_0008 (f388), pair_011 back back_moto_0001 (f54), pair_013 front front_moto_0003 (f400) — 4 cells. At the same roadside spot in three different videos, the box lands on a squatting man's torso/head-and-backpack.
- **Rickshaw cart / pedestrians:** pair_007 front front_moto_0005 (f129) — 2 cells (FRAG+FRONT); the box region contains a cycle-rickshaw cart and two standing men, no motorcycle.
- **Orphan tracks with no person:** pair_006 back back_orphan_0001 (f32) and back_orphan_0004 (f93) — boxes sit on **treetop foliage** at the top edge of the frame; pair_024 back back_orphan_0001 (f397) — box sits on a **building facade/balcony**. In each case the real road users are elsewhere in the frame and separately detected.

## 4. Corrections layer (`work/pass2/corrections.csv` — 63 operations)

Pass 1 files are untouched; corrections are a separate layer to be applied when the final annotation is built.

| Correction type | Rows | Source | Content |
|---|---|---|---|
| EXCLUDE_FP_TRACK | 16 tracks | VISUAL_SHEET 6 / FULLRES 10 | False-positive tracks flagged for exclusion from the final layer (9 sheet-ERROR cells + 15 fullres-ERROR cells, grouped by track) |
| CLIP_OFF_FRAME_BOX | 20 tracks | PROGRAMMATIC | Per-track rule marking the 185 off-frame track-frame boxes invalid (no replacement coordinates invented) |
| KEEP_FRAGMENT_TRACK | 19 tracks | VISUAL_SHEET | C6 single-hit tracks visually verified as real brief/partial motorcycles; retained |
| SET_CONSTANT_HELMET_STATE | 8 subtracks | FULLRES | Per-frame helmet state corrected to the verified constant state (5× HELMET, 3× NO_HELMET) |
| **Total** | **63** | PROGRAMMATIC 20 / VISUAL_SHEET 25 / FULLRES 18 | |

C6 disposition: 27 fragment tracks = **8 false positives excluded** (pair_007 front_moto_0005; pair_009 back_moto_0002/0003/0009/0010/0011; pair_011 back_moto_0001; pair_012 back_moto_0001) + **19 real retained**.

## 5. Unresolved items

**No unresolved audit cells remain** — all 258 inspected cells carry a final verdict (161 CORRECT, 24 ERROR, 73 FLIP_NOISE). Carried forward explicitly, by design:

1. The 185 off-frame track boxes are marked **invalid**, not re-estimated; the final layer must drop or clip them (correction type CLIP_OFF_FRAME_BOX).
2. pair_002 front_moto_0001_rider_s0 is the weakest-evidence FLIP resolution: the rider's head is ~50 px tall even at native resolution. The NO_HELMET call rests on consistent bare-hair appearance across the f0/f2/f4 crops; it is recorded as FLIP_NOISE with that caveat.
3. Frames and tracks **not** sampled by the visual pass were not visually verified (see §7) — absence of a correction is not a correctness guarantee.

## 6. Recurring observations

- **Auto-rickshaws are the dominant FP source.** 10 of the 16 excluded tracks are three-wheelers or rickshaw carts boxed as motorcycles, concentrated in pair_009 back (6 tracks: 0001/0002/0003/0009/0010/0011) — in that video essentially every "motorcycle" track after back_moto_0007 is suspect, while the one real motorcycle (back_moto_0004) is tracked correctly.
- **A squatting pedestrian is repeatedly classified as a motorcycle** (pair_009 back, pair_011 back, pair_013 front — same roadside location): head + shoulders + backpack at roadside reads as a motorcycle to the detector at ~90×110 px scale.
- **Orphan (head/person) false positives occur at the top edge of the frame** — treetop foliage (pair_006 back ×2) and building facades (pair_024 back) generate head-class detections that spawn orphan tracks with no person present.
- **Track fragmentation is mostly real.** 19 of 27 single-hit moto tracks are genuine brief/partial motorcycles (distant, entering, or leaving); only 8 are false positives. Fragment count alone is not an FP signal.
- **Rider-subtrack boxes drift when no head detection couples.** The subtrack box is tracker-propagated between observations; terminal boxes slide off the head — out of the frame entirely in pair_015 (head box exits right edge from f299) and onto scenery in pair_016/pair_023 end-of-track frames. This propagation is the mechanism behind most C4 off-frame findings and means late-track subtrack boxes must not be trusted as head locations.
- **Helmet flips are a small-head artefact.** All 73 flips occur where the head is small or the viewing angle changes; the head classifier alternates on an unchanging head. Aggregated subtrack state is reliable only where large-head observations exist.

## 7. Limitations

- **AI-assisted audit.** This audit was performed by an AI agent inspecting extracted frames, under the mission's AI-assisted protocol. It is **not** independent human verification, and the dataset must continue to be labelled **AI_ASSISTED_AUDITED**, never human-verified ground truth.
- **Contact-sheet resolution limit (mitigated).** The first visual pass worked from 620 px-wide sheet cells; all 88 cells it could not resolve (34% of the sample, including 100% of FLIP cells) were re-inspected at native 3840×2160 resolution before any verdict was recorded. No verdict in this report rests on sheet-scale evidence alone where the sheet pass declared uncertainty.
- **Coverage is not correctness.** The programmatic pass covered 100% of frames (15,241) but checks structure, not semantics. The visual passes covered 258 targeted cells plus 141 context/verification crops — a flagged-and-sampled subset, not every frame or every track. Error rates measured here apply to the inspected sample; uninspected annotations may contain further errors of the same kinds.
- **Single auditor.** Pass 2 verdicts come from one audit chain (programmatic → sheet → full-res). The independent re-audit is Pass 3's job; disagreements there should supersede this report.
- **Corrections are flags and rules, not re-annotation.** Excluded tracks and invalid boxes are marked in the corrections layer; geometric errors in retained boxes (loose/tight fits) were noted only where they changed a verdict and were not exhaustively re-measured.
