# Phase 12 — Annotation Final Audit Report (Pass 3)

**Audit type:** Pass 3, the final independent re-audit of the dual-camera helmet
annotation mission. This audit re-derived its evidence from the source videos and
the raw Pass 1 files. Pass 2 conclusions were not re-read as evidence; the only
Pass 2 artefact consumed as input is `work/pass2/corrections.csv`, i.e. the
correction layer under test.

**Dataset label:** `AI_ASSISTED_AUDITED`. This audit is an AI-assisted re-audit.
It is **not** independent human verification, and nothing in this report makes
the dataset human-verified ground truth.

**Working artefacts (VM):** `~/workspace/research_helmet/work/pass3/`
(`integrity_results.json`, `corrected_view_validation.json`,
`correction_verification.csv`, `fresh_sample_verdicts.csv`,
`fresh_sample_tally.json`, `extract_plan.json`, `rendered_manifest.json`,
rendered evidence images in `img/`).

**Scope of the audited data:** 25 pairs, 50 videos, 15,241 decoded frames
(front 6,557 + back 8,684), 44,311 Pass 1 detection boxes
(motorcycle 12,156 / person 22,350 / head 9,805).

---

## 1. Integrity of originals, metadata, and the Pass 1 release

Measured by `work/pass3/step1_integrity.py`:

| Check | Measured result |
|---|---|
| Source videos vs Phase 11 `pair_manifest.csv` SHA-256 | **50/50 match**, 0 missing, 0 mismatch |
| Phase 11 metadata files vs pristine `repo.tar.gz` copies | **6/6 byte-identical** |
| `work/pass1/` vs Kaggle `pass1_outputs.zip` (size + CRC32) | **300/300 zip-mapped files match**; `pass1_driver.log` also matches the zip copy (301 files accounted for; zip holds 326 entries incl. 25 directory entries) |

Conclusion: original videos, Phase 11 metadata, and the immutable Pass 1 release
were unchanged at the time of this audit.

## 2. Corrected-view validation (programmatic, full coverage)

Method: the 63 Pass 2 correction ops were applied as an in-memory view over the
raw Pass 1 files (Pass 1 was not modified) and structural validation was re-run
on the corrected view. Full results: `work/pass3/corrected_view_validation.json`.

**All 20 checks PASS:**

| # | Check | Measured |
|---|---|---|
| 1 | Frame coverage rows | 15,241 / 15,241 |
| 2 | Videos whose coverage count matches summary | 50 / 50 |
| 3 | Videos with contiguous frame indices | 50 / 50 |
| 4 | Videos with monotonic non-decreasing PTS (0 duplicate-PTS rows) | 50 / 50 |
| 5 | Detection frame records | 15,241 / 15,241 |
| 6 | Detection boxes | 44,311 / 44,311 |
| 7 | Frames where per-frame detection counts mismatch coverage columns | 0 |
| 8 | Invalid detection bboxes | 0 |
| 9 | Out-of-range detection confidences | 0 |
| 10 | Unknown track references | 0 |
| 11 | Off-frame track boxes in raw Pass 1 (>100 px beyond bounds) | 185 |
| 12 | Of those, on the 20 CLIP-listed tracks | 185 / 185 |
| 13 | Off-frame track boxes remaining in corrected view | **0** |
| 14 | EXCLUDE_FP_TRACK ops resolving to tracks with appearances | 16 / 16 (165 track-frame appearances removed in the corrected view) |
| 15 | Pairs whose association outcome counts match their summaries | 25 / 25 |
| 16 | Correction targets resolved (EXCLUDE + CLIP + KEEP) | 55 / 55 |
| 17 | Raw direct helmet-state flips | 73 |
| 18 | Of those, inside the 8 corrected subtracks | 73 / 73 |
| 19 | Flips outside the corrected subtracks | **0** |
| 20 | Corrected subtracks showing one constant state | 8 / 8 |

Corrected cross-camera association recount (results referencing excluded tracks
dropped: 11 moto + 5 orphan results removed):

| Level | Raw | Corrected |
|---|---|---|
| Moto LINKED | 12 | 11 |
| Moto UNMATCHED | 40 | 39 |
| Moto INSUFFICIENT_EVIDENCE | 27 | 19 |
| Moto AMBIGUOUS_ABSTAIN | 6 | 5 |
| Rider LINKED | 4 | 4 |
| Rider UNMATCHED | 6 | 4 |
| Rider INSUFFICIENT_EVIDENCE | 9 | 9 |
| Rider AMBIGUOUS_ABSTAIN | 1 | 1 |
| Orphan LINKED | 1 | 1 |
| Orphan UNMATCHED | 43 | 42 |
| Orphan INSUFFICIENT_EVIDENCE | 56 | 53 |
| Orphan AMBIGUOUS_ABSTAIN | 20 | 19 |

## 3. Correction spot-verification (visual, full resolution)

All 16 EXCLUDE_FP_TRACK ops and all 8 SET_CONSTANT_HELMET_STATE ops were
independently re-checked from source-video frames at full resolution (sequential
decode only; helmet states judged only at frames with a genuinely coupled head
detection, avoiding stale Kalman-propagated boxes). Per-op evidence:
`work/pass3/correction_verification.csv` (24 rows).

**Result: 24/24 CONFIRMED, 0 REJECTED, 0 UNCERTAIN.**

- EXCLUDE (16): every excluded track was confirmed to sit on a non-motorcycle /
  non-person target — enclosed auto-rickshaws (pair_009 back_moto_0001/0002/0003/
  0009/0010/0011, pair_012 back_moto_0001 on a rickshaw wheel/fork), squatting
  pedestrians (pair_009 back_moto_0008, pair_011 back_moto_0001, pair_013
  front_moto_0003), a pedal rickshaw cart (pair_007 front_moto_0005), and
  background orphan tracks on treetops / building facades (pair_002, pair_006 ×2,
  pair_023, pair_024).
- SET_CONSTANT (8): every corrected constant state was confirmed at coupled
  frames, including frames where the raw per-frame state contradicted the
  constant — HELMET ×5 (pair_010 back s0, pair_011 front s0, pair_012 front
  moto_0004 s0, pair_015 front moto_0002 s0, pair_016 front s0), NO_HELMET ×3
  (pair_002 front s0, pair_023 front s0, pair_024 back s0). Mechanism identified
  for two NO_HELMET subtracks: in pair_023 and pair_024 the rider carries a
  helmet **unworn** on the motorcycle (front / handlebar), the plausible source
  of the raw HELMET blips that the correction suppresses.

## 4. Fresh visual sample (visual, from source videos)

A new sample was built from frames **not** in the Pass 2 review plan: for each of
the 25 pairs, per camera, the mid-overlap frame plus one frame near a track start
and one near a track end (150 frames), plus one extra frame per camera for the
special-attention pairs (VFR / mixed-fps pairs 013–018 and near-frame pairs 004,
005, 008, 015; 9 unique pairs, +18 frames) = **168 frames**. Each frame was
rendered at full resolution with its corrected-view annotations and judged
against them (detections present/absent, box placement, helmet-state
plausibility). A frame is ERROR if any checked annotation element is defective;
missed detections were counted as errors only when the object is prominent —
sub-resolution distant misses are noted in the verdict notes, not counted.
Verdicts: `work/pass3/fresh_sample_verdicts.csv` (168/168 rows).

### 4.1 Overall tally and residual error rate

| Verdict | Frames | Share |
|---|---|---|
| CORRECT | 94 | 56.0% |
| ERROR | 66 | 39.3% |
| UNCERTAIN | 8 | 4.8% |

**Measured residual error rate: 66/168 = 39.3% of all sampled frames contain at
least one annotation defect; 66/160 = 41.3% of decided (non-UNCERTAIN) frames.**

Error types (one primary type per ERROR frame):

| Error type | Frames |
|---|---|
| RIDER_BOX_SPURIOUS (rider subtrack bbox on background / off-frame / tail after exit) | 37 |
| FP_DETECTION (false-positive detection or track: rickshaw-as-motorcycle, track on a pedestrian, person/head detection on background objects) | 20 |
| RIDER_BOX_MISPLACED (rider subtrack bbox on the wrong person — e.g. a pedestrian beside the bike — or offset off the rider's head) | 9 |

Layer decomposition of the 66 ERROR frames: in **46 frames the only defect is
rider subtrack bbox placement/association** — detections, motorcycle tracks,
person boxes, and helmet-state values in those frames were verified correct.
Only **20/168 frames (11.9%)** contain a defect in the primary
detection/track/person layer.

### 4.2 Per-group tallies

| Group | n | CORRECT | ERROR | UNCERTAIN | Error rate (of all) |
|---|---|---|---|---|---|
| VFR / mixed-fps pairs (013, 014, 015, 016, 017, 018) | 48 | 28 | 19 | 1 | 39.6% |
| Near-frame pairs (004, 005, 008, 015) | 32 | 18 | 14 | 0 | 43.8% |
| Special-attention union (9 pairs) | 72 | 42 | 29 | 1 | 40.3% |
| All other pairs | 96 | 52 | 37 | 7 | 38.5% |
| Front cameras | 84 | 37 | 42 | 5 | 50.0% |
| Back cameras | 84 | 57 | 24 | 3 | 28.6% |

The error rate is essentially uniform across pair groups: there is **no
VFR/near-frame-specific degradation** — the residual defects are a property of
the annotation layers, not of synchronisation or frame-rate mixing. Front-camera
frames err more often than back-camera frames (50.0% vs 28.6%), consistent with
front views carrying more rider-subtrack activity.

Per pair (n / CORRECT / ERROR / UNCERTAIN): pair_001 6/2/4/0 · pair_002 6/4/2/0 ·
pair_003 6/2/1/3 · pair_004 8/3/5/0 · pair_005 8/6/2/0 · pair_006 6/4/1/1 ·
pair_007 6/4/0/2 · pair_008 8/5/3/0 · pair_009 6/3/3/0 · pair_010 6/3/3/0 ·
pair_011 6/2/3/1 · pair_012 6/1/5/0 · pair_013 8/3/4/1 · pair_014 8/6/2/0 ·
pair_015 8/4/4/0 · pair_016 8/6/2/0 · pair_017 8/6/2/0 · pair_018 8/3/5/0 ·
pair_019 6/4/2/0 · pair_020 6/4/2/0 · pair_021 6/4/2/0 · pair_022 6/3/3/0 ·
pair_023 6/4/2/0 · pair_024 6/3/3/0 · pair_025 6/5/1/0.

### 4.3 Principal residual defect classes (measured in the fresh sample)

1. **Rider subtrack bounding boxes are unreliable (46/66 error frames).**
   Stale Kalman propagations sit on background (grass, trees, road surface,
   buildings, sky), remain stuck at one position across frames (pair_004
   front_moto_0002; pair_019 front_moto_0002), become degenerate (~4×3 px,
   pair_002 f293), go fully off-frame including negative coordinates
   (pair_003 back f433: [-1557.82, -1118.48, -1138.89, -739.59]), and persist as
   tails after the rider leaves. Phantom extra subtracks attach to pedestrians
   standing/walking beside a motorcycle (pair_018 front s1, pair_024 front s1)
   or to background (pair_023 front s1/s2). No Pass 2 correction covered rider
   subtrack boxes (CLIP covered motorcycle track boxes only). **Helmet-state
   values are far more reliable than subtrack box positions**: wherever a state
   could be verified at a coupled frame in this sample, it was correct,
   including all 8 corrected constants.
2. **False-positive motorcycle tracks on pedestrians that Pass 2 did not
   exclude (6 tracks found in this sample):** pair_011 back_moto_0003
   (squatting pedestrian; sibling back_moto_0001 was excluded as C014 for the
   same behaviour), pair_012 front_moto_0002 (seated roadside man), pair_013
   front_moto_0002 (squatting pedestrian; sibling front_moto_0003 was excluded
   as C015), pair_014 back_moto_0003 (squatting pedestrian), pair_015
   front_moto_0003 (squatting pedestrian). Plus one on a vehicle: pair_011
   front_moto_0002 (live track on a distant auto-rickshaw).
3. **Residual false-positive motorcycle detections on auto-rickshaws/mishuks**
   beyond the Pass 2 exclusions: pair_004 front f2 and back f198, pair_009 back
   f88 and f412, pair_012 front f235 and back f85, pair_014 back f78 (detection
   on a squatting pedestrian).
4. **Isolated detection-layer FPs on background objects:** person detection on
   a roadside sapling (pair_008 front f39), person detection on utility-pole
   posters and a head detection on a motorcycle windshield (pair_022 back f194).

Positive controls in the sample: the primary layer handled hard cases correctly
— a three-rider motorcycle (pair_007 back f236), a pedestrian carrying an unworn
helmet not misread as a rider (pair_010 front f2), carried helmets correctly not
counted as worn in at least 8 sampled frames, and rickshaws correctly left
unannotated in many frames.

## 5. Unresolved / uncertain items (explicit list)

Fresh-sample frames judged UNCERTAIN (never guessed; all involve objects at the
resolution floor):

1. pair_003 front f2 (track_start) — second claimed motorcycle detection on a
   tiny distant canopied vehicle; motorcycle vs auto-rickshaw unresolvable.
2. pair_003 front f231 (track_end) — same distant vehicle; class unresolvable.
3. pair_003 back f156 (mid_overlap) — two distant motorcycles visible but only
   one motorcycle detection claimed, hidden under a coincident person box; a
   missed detection on one bike can neither be confirmed nor excluded.
4. pair_006 back f66 (mid_overlap) — coincident moto+person boxes on a tiny
   distant figure group beside a rickshaw; presence of a motorcycle
   unresolvable.
5. pair_007 front f78 (mid_overlap) — second claimed track front_moto_0004 on a
   tiny distant canopied vehicle reading as a rickshaw cart (~35 px).
6. pair_007 front f2 (track_start) — same distant object as f78.
7. pair_011 back f98 (mid_overlap) — one moto + one person detection claimed;
   only candidates are an unresolvable roadside figure/board and a tiny distant
   motorcycle; placement can neither be confirmed nor refuted.
8. pair_013 front f401 (track_end) — single claimed motorcycle detection cannot
   be located at render scale (possibly coincident on the squatting man, whose
   person box is correct, or on a far-distant vehicle).

Additionally disclosed, not hidden: the 6 non-excluded FP tracks and the
residual FP detection classes in §4.3 remain present in the corrected dataset,
and further instances may exist in frames outside this 168-frame sample. The 19
KEEP_FRAGMENT_TRACK ops were validated programmatically (Step 2) but were not
part of the visual spot-verification set.

## 6. Annotation-mission completion-gate checklist

Each item marked from evidence produced or re-verified in this audit:

| Gate item | Status | Evidence |
|---|---|---|
| All pairs/videos processed | **PASS** | 25/25 pairs, 50/50 videos; coverage and hashes re-verified (§1, §2) |
| Every decoded frame in coverage manifest | **PASS** | 15,241/15,241 rows; 50/50 videos pass count, contiguity, and PTS checks (§2) |
| Passes 1–3 complete | **PASS** | Pass 1 immutable release (integrity §1); Pass 2 audit + 63-op corrections + `PHASE12_ANNOTATION_AUDIT_1_REPORT.md`; Pass 3 = this report |
| Corrections applied as a layer (Pass 1 untouched) | **PASS** | Corrected-view validation 20/20 (§2); 24/24 key ops visually re-confirmed (§3); Pass 1 byte-integrity proven against the Kaggle zip (§1) |
| Schema / ID validation | **PASS** (annotation-data level) | 0 unknown track references; per-frame detection counts reconcile with coverage on all 15,241 frames; association counts match summaries 25/25. Formal validation of the packaged `dual_camera_gt_v1.json` against `annotation_schema.json` belongs to the package build (below) |
| Reports + checksums | **FAIL — not yet built** | Pass 2 and Pass 3 reports exist. The final annotation package (`data/dual_camera_test_v1/annotations/`, incl. `checksums/annotation_checksums.sha256`) does **not** exist yet; it is a separate, still-pending build step outside Pass 3 |
| Originals / Phase 11 metadata unchanged | **PASS** | 50/50 video SHA-256 match; 6/6 metadata files byte-identical (§1) |
| Unresolved cases documented | **PASS** | §5 lists all 8 UNCERTAIN sample frames and the residual defect classes, including 6 newly found FP tracks |

**Gate outcome: the audit itself is complete, but the annotation mission's
completion gate does NOT fully pass yet** — the final package + checksums item
remains open. Separately, the fresh sample measures a non-trivial residual
defect profile (§4): gate passage on process completeness must not be read as
zero annotation defects.

## 7. Limitations

- **AI-assisted audit.** Pass 3 was executed by an AI agent re-inspecting
  renders and full-resolution crops of the source videos. It is a second,
  independent-of-Pass-2 AI pass — it is not human verification. The dataset
  remains labelled `AI_ASSISTED_AUDITED`.
- **Sampling-based visual component.** The Pass 3 visual sample covers 168 of
  15,241 frames (1.1%), drawn by a structured design (mid-overlap / track-start /
  track-end per camera, extras for special pairs), not by random sampling;
  frame-level error rates therefore describe the sampled strata and should not
  be extrapolated to a per-box accuracy figure. The programmatic component
  (§1–§2) covers 100% of frames, detections, tracks, and correction ops.
- **Frame-level metric.** A frame with one defective element counts once as
  ERROR; the 39.3% figure is not a box-level error rate. The dominant term is
  the rider subtrack bbox layer (46/66), which downstream users should treat as
  unreliable; helmet-state values and the primary detection/track layer
  measured materially better (state values verified correct at every coupled
  check in this audit; detection-layer defects in 11.9% of sampled frames).
- **Resolution floor.** Distant objects (~25–55 px) cannot always be resolved
  even at full resolution; such cases were recorded as UNCERTAIN with reasons
  (§5) rather than guessed.
- **Known residual defects remain in the dataset** (§4.3): rider subtrack box
  placement was not corrected by any pass; 6 FP motorcycle tracks found in
  Pass 3 were not in the Pass 2 exclusion list; auto-rickshaw FP detections
  persist at the detection layer. Any future correction round should target
  these classes explicitly.
- Processing coverage, annotation coverage, audit coverage, and correctness are
  distinct quantities; this report states each separately and claims no
  100% accuracy anywhere.

---

*Pass 3 executed 2026-11-09/10 (VM clock). Auditor: AI agent (Pass 3 session),
independent of the Pass 2 audit session. Evidence retained under
`work/pass3/` (scripts, JSON results, verdict CSVs, rendered images).*
