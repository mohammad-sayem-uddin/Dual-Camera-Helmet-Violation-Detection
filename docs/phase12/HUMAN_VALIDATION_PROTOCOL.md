# HUMAN VALIDATION PROTOCOL — Dual-Camera Helmet Dataset

**Document status: PROTOCOL ONLY — DESIGNED, NOT EXECUTED.**
**Version:** 1.0 (2026-11-12) · **Applies to:** frozen release
`dual_camera_gt_v1.0_ai_assisted_audited`
(`data/dual_camera_test_v1/releases/dual_camera_gt_v1.0_ai_assisted_audited/`)

> **No human verification of this dataset exists.** This document specifies how an
> independent human validation *would* be performed. Until the protocol is actually
> executed and its completion record (§9) is filled in with reviewer identities,
> dates and measured metrics, the dataset remains labelled **`AI_ASSISTED_AUDITED`**
> with `human_verified = false`, and no document, paper or presentation may describe
> any part of it as human-verified. See §8 for the binding statement.

---

## 0. Corpus facts this protocol is built on (measured, not assumed)

All counts below were read directly from the frozen release files and the Stage K
evaluation artifacts (`work/stageK/stagek_results.json`) on 2026-11-12. Sampling
strata in §2 are defined over exactly these numbers.

| Quantity | Value | Source |
|---|---|---|
| Camera pairs / videos / decoded frames | 25 / 50 / 15,241 (front 6,557 + back 8,684) | release README |
| Global events | **70** = DRIVER_NO_HELMET 27, BOTH_NO_HELMET 10, PILLION_NO_HELMET 3, NO_VIOLATION 6, INSUFFICIENT_VISUAL_EVIDENCE 24 | `dual_camera_gt_events.csv` |
| GT motorcycle tracks retained | **82** (front 42, back 40); pipeline produced 100, the 18 extras are exactly the audit-excluded false-positive set | Stage K `detection_motorcycle_tracklevel` |
| Orphan (uncoupled) tracks retained | **125** (front 71, back 54) | release README |
| Decided rider tracks (GT helmet state HELMET or NO_HELMET) | **209** | Stage K `helmet_track_level` |
| — pipeline agrees with release | **134** (GT HELMET 38, GT NO_HELMET 96) | Stage K confusion |
| — pipeline track verdict UNCERTAIN (abstained) | **73** (GT HELMET 18, GT NO_HELMET 55) | Stage K confusion |
| — hard cross-disagreements | **2** (1× GT helmet → pred NO_HELMET; 1× GT no_helmet → pred HELMET) | Stage K confusion |
| Decided GT head observations | **6,753** (HELMET 1,365 / NO_HELMET 5,388); 6 further observations excluded as `occluded_unknown` (pair_004 donning windows) | Stage K `helmet_observation_level` |
| Rider track-frame rows (moto subtracks) | 12,458 by corrected state: HELMET 1,825 / NO_HELMET 5,686 / UNCERTAIN 4,947 | base package accounting |
| Rider-frame validity layer | 14,249 rows (moto + orphan basis); only **6,727** `spatial_use_valid`; all 760 pair_002 rows spatially invalid | `rider_frame_validity.csv` |
| Low-evidence subtracks (≤2 coupled observations, ≥30 rows) | **30** subtracks, 3,907 rows, flagged `LOW_EVIDENCE_SUBTRACK` | limitations doc §3 |
| Identity links | **202** rows = motorcycle 70, rider 17, orphan_rider 115 | `cross_camera_identity_links.csv` |
| Event link status | LINKED_BOTH_VIEWS **10**, FRONT_VIEW_ONLY **29**, BACK_VIEW_ONLY **27**, AMBIGUOUS_HUMAN_UNCERTAIN **4** | links CSV (event rows) |
| Association behaviour (Stage K) | moto: pred LINKED 12 vs GT 10 (2 FP links: pair_009 F1×B1, pair_015 F3×B1), abstention 33/85 = 38.8%; orphan: abstention 76/120 = 63.3%, GT linked = 1 | Stage K `association_*` |
| Release uncertainty register | **178** rows — by source: CORRECTED_VIEW_DERIVATION 83, PASS3_FRESH_VISUAL_SAMPLE 54, EVENT_DERIVATION 24, STAGE_D_FINAL_AUDIT 12, STAGE_C_CORRECTION 5 | `annotation_uncertainty.csv` |
| Sync uncertainty relevant to event timing | worst labelled bound ε_sync + \|drift\| = 114 ms (pair_008); drift pairs 004 +36 ms, 005 +70 ms, 008 −76 ms, 015 +74 ms; no exposure sync / genlock exists | Stage F/L reports |
| Prior AI-only visual evidence (not human) | Stage D coupled-stratum sample: helmet-state error 3.3% of decided frames, placement error 1.6% | release README / Stage D |

---

## 1. Purpose and scope

### 1.1 Purpose

To obtain the first **human** reference judgements for a stratified sample of the
AI-assisted annotations, so that (a) the accuracy of the AI labels can be estimated
against human judgement with confidence intervals, per stratum; (b) the known weak
points of the programme — helmet track-level aggregation (64.1% pipeline agreement),
association abstention, rider-box placement, event boundaries — can be confirmed,
refuted or bounded by people rather than by further AI passes; and (c) any future
claim stronger than `AI_ASSISTED_AUDITED` has a documented, auditable basis.

### 1.2 What executing this protocol WOULD establish

- An unbiased (within-stratum) estimate of AI label error rates for: track-level
  helmet state, driver/pillion role attribution, cross-camera identity outcomes,
  event type and event onset/end timing, and orphan-vs-rider classification — each
  reported with a 95% confidence interval and the sampling fraction used.
- Adjudicated human reference labels for exactly the items in the frozen validation
  sample (§6.1), usable as a held-out test set for any future system version.
- A per-stratum pass/fail verdict against the pre-registered thresholds in §5.3.

### 1.3 What it would NOT establish

- **Nothing about unreviewed items.** 135 of 209 decided rider tracks are reviewed;
  the other 74 (the unsampled agreeing tracks) remain AI-labelled only, as do all
  unreviewed links, frames and observations.
- **No field or deployment accuracy.** The corpus is 25 pairs from one capture
  setting; agreement with human reviewers on this corpus says nothing about other
  cameras, locations, lighting or rider populations.
- **No rescue of non-existent metric families.** Person-detection ground truth,
  frame-level motorcycle boxes and HOTA remain undefined in this release; human
  review under this protocol does not create them.
- **No retroactive change to the frozen release** (§6.4).

### 1.4 Labelling rule (binding)

- The dataset as a whole keeps the label **`AI_ASSISTED_AUDITED`** permanently for
  v1.0, whatever the outcome.
- Only the specific items in the validation sample, and only in strata that pass
  the §5.3 thresholds, may be designated **`HUMAN_VERIFIED_SAMPLE`** — always with
  the qualifier "sample", the stratum name, the sample size and the protocol
  version attached (e.g. "helmet track states: HUMAN_VERIFIED_SAMPLE, n = 135/209,
  HV protocol v1.0").
- A stratum that fails its threshold is reported as failed. It is not re-sampled
  until it passes (§5.4).

---

## 2. Sampling methodology

### 2.0 Design principles

- **Census where the claim is rare, high-stakes or known-weak; random sampling
  only where the population is large and homogeneous** (the agreeing majority).
- **Stratification variables:** pipeline agreement status, GT helmet state, camera
  (front/back), pair sync stratum (near-frame pairs 004/005/008/015 and VFR
  streams oversampled implicitly — they concentrate in the disagreement and
  low-evidence strata), and event type.
- **Randomisation:** all random draws use a recorded seed (the protocol coordinator
  fixes `seed = 20261112` at manifest build) applied to the sorted item IDs of each
  stratum; the draw script and its output list are part of the frozen manifest
  (§6.1), so the sample is reproducible and auditable.
- **Decision unit discipline:** each stratum has one decision unit (track, link,
  event, frame item). Reviewers never judge "the dataset in general".
- **Keyframes:** for track-level items the reviewer is shown up to 5 *coupled*
  keyframes (frames where the track has a genuine same-frame detection:
  first / 25% / median / 75% / last coupled frame) plus a scrubbable ±1 s clip
  around the median keyframe. Keyframe state judgements double as an
  observation-level sample (≈540 frame judgements in Stratum A).

### 2.1 Stratum A — helmet classification (track level) — 135 tracks

Universe: the **209** decided rider tracks (Stage K basis).

| Sub-stratum | Definition | Population | Reviewed | Rule |
|---|---|---|---|---|
| A1 | Pipeline verdict UNCERTAIN (abstained) | 73 | **73** | Census — this is the known failure mode; includes every pair_002 subtrack |
| A2 | Hard cross-disagreement | 2 | **2** | Census |
| A3 | Pipeline agrees with release | 134 | **60** | Random sample, stratified by GT state: 17 of 38 HELMET-agreeing, 43 of 96 NO_HELMET-agreeing (proportional) |

Sample-size justification for A3: estimating the agreeing stratum's error rate
with a ±10 percentage-point worst-case margin at 95% confidence needs
n₀ = 1.96² · 0.25 / 0.10² ≈ 96 from an infinite population; with the finite
population correction for N = 134, n = 96 / (1 + 95/134) ≈ 56. We adopt **60**,
which also gives the rule-of-three guarantee: if reviewers find 0 errors in 60,
the 95% upper bound on the agreeing stratum's true error rate is ≈ 3/60 = 5.0%.

Reviewer task per track: judge the rider's helmet state from the keyframe set as
HELMET / NO_HELMET / CANNOT_TELL (with confidence sure/unsure and a comment).
CANNOT_TELL is a legitimate verdict and is never forced into a state.

### 2.2 Stratum B — rider identity and driver/pillion distinction

| Sub-stratum | Definition | Population | Reviewed | Rule |
|---|---|---|---|---|
| B1 | Role attribution on decided events (driver vs pillion vs single rider; correct rider count per motorcycle) | 46 events (27 DRIVER + 10 BOTH + 3 PILLION + 6 NO_VIOLATION) | **46** | Census, executed inside the Stratum D event sessions (same sitting, separate decision fields). The known pair_002 role error (GT BOTH_NO_HELMET vs pipeline PILLION_NO_HELMET) is in this set by construction |
| B2 | Orphan-vs-rider classification: is this track truly *not* a motorcycle rider? | 125 orphan tracks (front 71, back 54) | **30** | Random (≈24%); if 0 errors, 95% upper bound on misclassification ≈ 3/30 = 10% |

### 2.3 Stratum C — cross-camera motorcycle identity — 85 link units

| Sub-stratum | Definition | Population | Reviewed | Rule |
|---|---|---|---|---|
| C1 | LINKED_BOTH_VIEWS motorcycle outcomes | 10 | **10** | Census — every positive identity claim is checked |
| C2 | AMBIGUOUS_HUMAN_UNCERTAIN event outcomes | 4 | **4** | Census |
| C3 | FRONT_VIEW_ONLY outcomes | 29 | **15** | Random ≈50%; reviewer must also scan the back camera's overlap window (negative verification — is a counterpart really absent?) |
| C4 | BACK_VIEW_ONLY outcomes | 27 | **14** | Random ≈50%; same negative verification against the front camera |
| C5 | Rider-level links | 17 | **17** | Census (small n) |
| C6 | Orphan-rider link outcomes (mostly UNMATCHED / INSUFFICIENT_EVIDENCE) | 115 | **25** | Random ≈22%; if 0 errors, 95% upper bound ≈ 3/25 = 12% |

Pooled negative-verification bound: C3 + C4 contribute 29 sampled single-view
outcomes; 0 missed links among them bounds the missed-link rate at ≈ 3/29 ≈ 10%
(95%). The 2 known pipeline FP links (pair_009, pair_015) are *pipeline* outputs,
not release claims; the release-side claims checked here are the 10 GT links.

Reviewer task per unit: SAME motorcycle/rider or DIFFERENT, or CANNOT_TELL,
judged from both cameras' track keyframes and the transit window — never from the
AI's association score, which is hidden (§3.3).

### 2.4 Stratum D — violation events and boundaries — 70 events (census)

All **70** events are reviewed. A census is justified: 70 is the headline claim
set of the entire dataset, it is small, and it includes the 24
INSUFFICIENT_VISUAL_EVIDENCE events whose correctness consists precisely in a
human agreeing that the evidence is insufficient.

Reviewer task per event, on the back-camera (reference) timeline:
1. Event type: DRIVER_NO_HELMET / PILLION_NO_HELMET / BOTH_NO_HELMET /
   NO_VIOLATION / INSUFFICIENT_VISUAL_EVIDENCE / NO_EVENT_HERE.
2. Onset and end timestamps of the violation-relevant interval, marked
   independently of the stored window.
3. Stratum B1 role fields (for the 46 decided events).

**Boundary tolerance (pre-registered):** primary tolerance **±0.5 s**; sensitivity
analysis at **±0.25 s**. Both exceed the worst labelled sync bound in the corpus
(114 ms, pair_008) by more than 2× and 4× respectively, so a boundary disagreement
beyond tolerance cannot be explained away by synchronisation error. For the
near-frame pairs (004/005/008/015) timings are additionally interpreted against
their per-pair ε_sync + |drift| bounds (up to 114 ms), which are reported
alongside, never subtracted.

### 2.5 Stratum E — difficult and uncertain cases (oversampled by design)

| Sub-stratum | Definition | Population | Reviewed | Rule |
|---|---|---|---|---|
| E1 | EVENT_DERIVATION uncertainty rows | 24 rows | **24** | Census; reviewed inside Stratum D sessions, scored separately |
| E2 | Disclosed residual defects from Stage C / Stage D (incl. SD-F1's 4 state errors, SD-F2's 2 background boxes, SD-F3's off-frame orphan box) | 17 rows (5 + 12) | **17** | Census — verifies the disclosure register itself is accurate |
| E3 | LOW_EVIDENCE_SUBTRACKs (≤2 coupled observations) | 30 subtracks | **30** | Census of those not already drawn in Stratum A (deduplicated at manifest build; budgeted at the full 30) |
| E4 | pair_004 donning-transition orphans (front_orphan_0001, front_orphan_0003) + their 6 `occluded_unknown` observations | 2 tracks / 6 obs | **all** | Mandatory: reviewer either confirms "no constant state honestly assignable" or marks explicit transition boundaries |
| E5 | Spot-check of remaining frame-level uncertainty rows (PASS3_FRESH_VISUAL_SAMPLE 54 + CORRECTED_VIEW_DERIVATION 83) | 137 rows | **30** | Random |

### 2.6 Totals — review burden

Decision units per reviewer (before deduplication of E1/E3 overlaps, which only
reduces the load):

| Stratum | Units |
|---|---|
| A helmet tracks | 135 |
| B2 orphan classification | 30 |
| C link units | 85 |
| D events (incl. B1 roles + E1 flags) | 70 |
| E2 + E3 + E4 + E5 items | 79 |
| **Total** | **≈ 399 decision units** |

Frame exhibits: ≈ 540 Stratum A keyframes + ≈ 340 link exhibits (85 units × ~4
frames) + 70 event clips with boundary frames + ≈ 160 Stratum E items'
frames ≈ **1,000–1,200 frame exhibits** per reviewer pass.

### 2.7 Reviewer-hours (arithmetic shown)

Per-reviewer time, at measured task rates (state/role judgement ≈ 1.5–2.5 min per
unit including clip scrub; link verification across two cameras ≈ 2–5 min;
event typing + two boundary marks ≈ 5 min):

| Block | Arithmetic | Hours |
|---|---|---|
| Calibration set (§3.4) | 40 items, one-off | 1.5 |
| A | 135 tracks × 2.5 min | 5.6 |
| B2 | 30 × 1.5 min | 0.75 |
| C | (14 linked/ambiguous × 5 min) + (29 single-view × 4 min) + (17 rider × 3 min) + (25 orphan × 2 min) = 70 + 116 + 51 + 50 = 287 min | 4.8 |
| D | 70 events × 5 min (includes B1 role fields) | 5.8 |
| E | (17 × 3 min) + (30 × 2.5 min) + (2 tracks, 10 min) + (30 × 2 min) = 51 + 75 + 10 + 60 = 196 min | 3.3 |
| **Per reviewer** | | **≈ 21.8 ≈ 22 h** |

Programme total: 2 independent reviewers × 22 h ≈ **44 h**; adjudication
(§4: expect ≈ 15% of ≈ 400 units ≈ 60 contested items × 6 min ≈ 6 h, plus
sign-off ≈ 1 h) ≈ **7 h**; coordination (manifest build, exports, metrics,
report) ≈ **4 h**. **All-in ≈ 55 person-hours** — roughly 3 elapsed days with
two reviewers working in parallel, plus adjudication.

---

## 3. Reviewer plan

### 3.1 Roles (minimum staffing)

- **Reviewer 1 and Reviewer 2** — two fully independent reviewers. Every item in
  the sample is judged by both, independently, with no communication during the
  review phase.
- **Adjudicator (Reviewer 3)** — resolves disagreements (§4), owns the calibration
  answer key, and independently rates a 10% overlap subset (≈ 40 units drawn
  across strata) so that both Cohen's and Fleiss' kappa can be reported (§5.1).
- **Coordinator** (may be the adjudicator's team, never Reviewer 1/2 for their own
  items) — builds the frozen manifest, runs the review app, computes metrics.
  The coordinator does not judge sample items.

### 3.2 Qualifications and independence

- Demonstrated ability to identify motorcycle helmets (vs caps, hats, hoods —
  which the annotation taxonomy explicitly forbids calling helmets without
  evidence), rider seating positions, and motorcycle identity across viewpoints;
  verified on the calibration set (§3.4), not by self-declaration.
- **Independence:** reviewers must not have directed, prompted or set policy for
  the AI annotation/audit passes that produced this release, and must have no
  stake in the validation passing. (In this programme the annotation operators
  were AI passes working under a mission brief; anyone who wrote or approved that
  brief's annotation decisions is not independent for this purpose.)
- Reviewers are identified by name or persistent pseudonymous ID in the
  completion record (§9) — anonymous "a human checked it" claims are not
  acceptable evidence.

### 3.3 Blinding (non-negotiable)

- Reviewers see **raw frames and clips only**. AI labels, AI bounding boxes,
  association scores, pipeline verdicts and the release's event windows are
  **hidden** until the reviewer's own decision for that item is submitted and
  locked. The review UI (§3.5) enforces this technically, not by instruction.
- Reviewers do not see each other's decisions, the stratum a item belongs to, or
  whether an item is a census "known-hard" case vs a random control. Item order
  is shuffled per reviewer (recorded seeds) so strata cannot be inferred from
  ordering.
- Unblinding happens only at adjudication (§4) and analysis (§5), after both
  reviewers' full decision sets are locked and exported.

### 3.4 Training / calibration set

- **40 items** drawn from the corpus *outside* the validation sample (unsampled
  agreeing tracks, non-sampled orphan tracks, and event clips from pairs not
  dominating the sample), spanning every decision type in §2, including at least
  4 deliberate CANNOT_TELL / insufficient-evidence cases and 2 cross-camera
  negative cases.
- The adjudicator establishes the calibration answer key beforehand, with written
  rationales, and freezes it (hash recorded in the manifest).
- Each reviewer must reach **Cohen's κ ≥ 0.70 against the calibration key**
  before starting the real sample. A reviewer below threshold gets one retraining
  round with the rationales; a second failure replaces the reviewer. Calibration
  performance is reported in the final validation report.

### 3.5 Review UI requirements

The project already operates a review-app pattern (built for the WeatherGuard
NHR review): a local, stdlib-only server + single-page UI, one item at a time,
with a decisions database that starts empty and exports to CSV. The helmet
validation app follows the same pattern, with these requirements:

- **One item at a time**; no grid of thumbnails with labels.
- **Overlay toggle:** the AI annotation overlay exists only as a *post-decision*
  comparison view; it is disabled (not merely off by default) until the item's
  decision is locked.
- **Decision + confidence + comment** on every item: decision buttons per
  stratum type, confidence (sure / unsure), free-text comment, and an explicit
  CANNOT_TELL option everywhere.
- **Clip scrubbing** for event and link items (±1 s around keyframes minimum;
  full overlap window for link items), frame-step controls, zoom/pan for
  long-range frames.
- **Resume support:** reopening starts at the first unreviewed item; decisions
  are append-only in the reviewer's own database; no reviewer can see or edit
  another reviewer's database.
- **Exports:** per-reviewer CSV export (item ID, decision, confidence, comment,
  timestamp, time-on-item) — the raw material for §5 metrics. Exports are hashed
  into the completion record.

---

## 4. Adjudication process

1. **Lock and compare.** After both reviewers finish, the coordinator exports
   both decision sets and computes the raw disagreement list per stratum. No
   decision may be edited after this point by its author.
2. **Disagreement classes.** (a) State/type disagreements (HELMET vs NO_HELMET,
   event-type mismatches, SAME vs DIFFERENT links); (b) boundary disagreements
   (onset or end differing by more than the §2.4 primary tolerance ±0.5 s);
   (c) CANNOT_TELL vs a definite judgement — treated as a disagreement, not as
   an error by either party.
3. **Consensus rule.** Where the two reviewers agree, their label is the human
   reference label. Where they disagree, the **adjudicator decides**, viewing
   the item blind first (own judgement recorded), then with both reviewers'
   comments visible. The adjudicator's decision is final for that item.
4. **Three-way splits.** If the adjudicator's blind judgement differs from both
   reviewers, the item is escalated to a joint session (all three, item shown
   without any prior labels); majority decides; if no majority exists the item
   is recorded as HUMAN_UNRESOLVED and excluded from accuracy denominators
   (reported, never silently dropped).
5. **Boundary adjudication.** For onset/end, if the two reviewers' marks are
   within ±0.5 s, the reference boundary is their midpoint. Beyond tolerance,
   the adjudicator marks independently and that mark is the reference; the
   original spread is kept for the timing-agreement statistics (§5.2).
6. **Recording corrections.** Any item where the adjudicated human label differs
   from the release label is logged as a **human-validation finding** in a new
   file, `validation/human_validation_findings.csv`, with provenance
   `HUMAN_VALIDATION` per row (item ID, release label, Reviewer 1, Reviewer 2,
   adjudicated label, rationale). **The frozen release is never edited**
   (§6.4). Findings are inputs to a possible future v1.1 — nothing else.

---

## 5. Agreement metrics and pre-registered thresholds

All thresholds in §5.3 are fixed by this protocol version **before** any review
begins. Changing them after seeing results invalidates the validation.

### 5.1 Human inter-rater agreement (is the human reference itself trustworthy?)

- **Cohen's κ** between Reviewer 1 and Reviewer 2, computed per decision family
  (helmet track state, event type, link outcome, orphan classification) and
  pooled, on pre-adjudication decisions.
- **Fleiss' κ** over the ≈ 40-unit triple-rated overlap subset (both reviewers +
  adjudicator's blind ratings).
- Raw percent agreement reported alongside κ (κ alone hides prevalence effects;
  e.g. the NO_HELMET-heavy helmet strata).

### 5.2 AI-vs-human accuracy (per stratum, against adjudicated labels)

- **Helmet (Stratum A):** per-class precision/recall/F1 of the release track
  label vs the adjudicated human label; the pipeline's abstention (73 tracks) is
  scored as its own outcome — abstentions are never counted as correct, matching
  the Stage K convention. Keyframe-level judgements additionally give an
  observation-level P/R on ≈ 540 frames.
- **Roles (B1):** driver/pillion attribution accuracy over the 46 decided events;
  rider-count accuracy.
- **Links (Stratum C):** per-outcome accuracy; LINKED precision (of the 10 census
  links, how many humans confirm); missed-link rate on C3/C4 with its §2.3
  confidence bound; orphan-outcome accuracy on C6.
- **Events (Stratum D):** event-type agreement rate; **timing agreement** —
  mean and median absolute onset error and end error of the stored windows vs
  adjudicated boundaries, and the percentage of boundaries within ±0.25 s and
  ±0.5 s. For near-frame pairs the per-pair sync bound (≤ 114 ms) is reported
  next to the timing errors, never subtracted from them.
- All proportions reported with Wilson 95% confidence intervals, **per stratum**
  — pooled-only reporting is prohibited (it would let the large agreeing stratum
  drown the failure strata).

### 5.3 Pre-registered acceptance thresholds

| Check | Threshold |
|---|---|
| Human reliability gate: reviewer κ (helmet state) | ≥ 0.75 to proceed; 0.60–0.75 → adjudicate everything in that family and flag the family as low-reliability; < 0.60 → human labels in that family are **not** a valid reference: stop, retrain/replace, redo that family |
| Helmet track accuracy (release vs adjudicated, decided tracks) | ≥ 0.90 point estimate **and** lower 95% CI ≥ 0.85 |
| Helmet per-class recall | HELMET ≥ 0.90 and NO_HELMET ≥ 0.90 (a missed helmet and a phantom helmet are equally disqualifying) |
| Event-type agreement | ≥ 0.90 over the 70 census events |
| Event boundary timing | MAE ≤ 0.25 s for onset and for end; ≥ 90% of boundaries within ±0.5 s |
| LINKED precision | ≥ 0.90 (≥ 9 of the 10 census links confirmed) |
| Missed-link rate (C3/C4) | ≤ 10% observed, consistent with the §2.3 bound |
| Role attribution (B1) | ≥ 0.90 over the 46 decided events |

### 5.4 Consequences of failure

- A stratum that meets its thresholds earns the §1.4 `HUMAN_VERIFIED_SAMPLE`
  designation **for the reviewed items only**.
- A stratum that fails stays `AI_ASSISTED_AUDITED`; the failure, its measured
  rate and its CI are published in the validation report exactly like a pass.
  **There is no re-sampling until pass**: a failed stratum may be re-validated
  only under a new protocol version, on a fresh sample, after a documented cause
  analysis — and both attempts are reported.
- If the human reliability gate fails irrecoverably, the correct conclusion is
  that *this validation* failed, not that the AI labels are wrong; the dataset
  remains AI-labelled and the attempt is recorded in §9 of the executed report.

---

## 6. Separation and governance

### 6.1 Frozen sample manifest (before any review)

Before any reviewer sees any item, the coordinator builds
`validation/sample_manifest.csv` listing every sampled item (stratum, item ID,
pair, camera, track/event/link reference, keyframe frame indices), the draw
script and seeds, the calibration key hash, and this protocol's version; the
manifest is **SHA-256 hashed and the hash recorded** in the completion record
skeleton (§9). Any later change to the sample requires a new protocol version
and a new manifest — mid-review sample edits are prohibited.

### 6.2 Development-data separation

- The validation sample is **held out**: no item in it may be used to tune,
  threshold, prompt, select checkpoints for, or otherwise develop the annotation
  pipeline, the trackers, the association engine or any successor model, from
  manifest freeze until the validation report is published — and thereafter only
  as a *test* set, never as training/tuning data, with that status recorded
  wherever the set is referenced.
- Reviewers and adjudicators do not feed preliminary impressions back to any
  development work during the review; the only channel is the final report.

### 6.3 Provenance of human labels

Human decisions live in a **separate validation store** (`validation/` directory:
manifest, per-reviewer exports, findings CSV, metrics, report). Every human
label carries provenance `HUMAN_VALIDATION`, reviewer ID, timestamp and protocol
version. Human labels are never merged into the frozen release files.

### 6.4 The frozen release is never silently edited

`dual_camera_gt_v1.0_ai_assisted_audited` and its checksums remain bit-identical
forever. If, after a completed validation, the project chooses to adopt
adjudicated corrections, it must issue a **new release** (e.g.
`dual_camera_gt_v1.1_…`) as a separate directory with its own checksums, release
notes diffing every changed item against v1.0 and citing the validation findings
file as the source. Overwriting v1.0, in whole or in part, is prohibited.

### 6.5 Reporting

The executed validation produces `validation/HUMAN_VALIDATION_REPORT.md`:
per-stratum metrics with CIs (§5.2), inter-rater statistics (§5.1), threshold
verdicts (§5.3), the full findings list, calibration results, and the completed
§9 record. Negative and inconclusive results are reported with the same
prominence as passes.

---

## 7. Ethics and practical notes

- **Faces and bodies.** The footage shows identifiable riders and pedestrians.
  Review is a legitimate research use, but: review only on access-controlled,
  project-owned machines; no screenshots, exports or frame grabs may leave the
  project store; any frame reproduced in a report or publication must have faces
  blurred and must come from the already-disclosed audit material where possible.
- **License plates.** The release deliberately never assessed plates
  (`NOT_ASSESSED`, no plate text anywhere). Reviewers must **not** transcribe
  plate text into comments or findings; if a plate is legible in a reviewed
  frame, the item is flagged `PLATE_VISIBLE` and nothing more is recorded.
- **Access control.** Each reviewer gets an individual account/database; access
  is logged; access ends when the review closes. The validation store inherits
  the project's storage and backup rules; reviewer exports contain decisions,
  never raw personal data beyond the item references needed to audit them.
- **Reviewer welfare/practicalities.** Sessions should be capped at ~2 h of
  continuous judging (fatigue degrades κ); the app supports resume, so progress
  is never lost. Reviewers may decline any item (recorded as DECLINED, replaced
  by the adjudicator's blind judgement at adjudication, and counted).

---

## 8. Explicit statement of non-execution

> **This protocol has NOT been executed.** As of 2026-11-12, **no human being has
> reviewed, verified or validated any annotation** in
> `dual_camera_gt_v1.0_ai_assisted_audited` under this or any other protocol. All
> prior audits of this dataset (Passes 2–3, Stages B–D) were AI passes. The
> dataset's label is `AI_ASSISTED_AUDITED` and its `human_verified` field is
> `false`, and both remain so until a completed execution of this protocol is
> recorded per §9 — with reviewer identities, dates, the frozen manifest hash,
> and the measured metrics — in a `validation/HUMAN_VALIDATION_REPORT.md`.
> Any use of the phrase "human-verified" for this dataset before such a record
> exists is false, regardless of context.

---

## 9. Completion record (to be filled in ONLY when the protocol is executed)

| Field | Value |
|---|---|
| Protocol version executed | |
| Sample manifest SHA-256 | |
| Calibration key SHA-256 | |
| Reviewer 1 (name/ID, qualification basis) | |
| Reviewer 2 (name/ID, qualification basis) | |
| Adjudicator (name/ID) | |
| Review dates (start – end, per reviewer) | |
| Calibration κ per reviewer | |
| Items reviewed (per stratum, actual) | |
| Inter-rater κ (per family + pooled) / Fleiss' κ (overlap subset) | |
| Per-stratum AI-vs-human metrics with 95% CIs | |
| Threshold verdicts (§5.3, per check) | |
| Findings file (`validation/human_validation_findings.csv`) SHA-256 | |
| Strata granted `HUMAN_VERIFIED_SAMPLE` (with n) | |
| Strata failed / inconclusive | |
| v1.1 issued? (release name + checksums, or "no") | |
| Validation report path | |

---

## Appendix — execution checklist (for whoever runs this protocol)

1. Appoint coordinator, 2 reviewers, adjudicator; record identities (§9 skeleton).
2. Build calibration set + answer key; freeze and hash.
3. Build `validation/sample_manifest.csv` with seed 20261112 draws per §2; hash it.
4. Calibrate reviewers (κ ≥ 0.70 gate, §3.4).
5. Review phase: blind, independent, shuffled order, ≤ 2 h sessions.
6. Lock and export both decision sets; hash exports.
7. Adjudicate per §4; record findings with `HUMAN_VALIDATION` provenance.
8. Compute §5 metrics per stratum; apply §5.3 thresholds; write
   `validation/HUMAN_VALIDATION_REPORT.md`; complete §9.
9. Apply §1.4 designations exactly as earned — nothing more.
