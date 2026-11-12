# Post-Annotation Stage K — Evaluation of the Integrated Pipeline vs the Frozen Release

**Status: COMPLETE (2026-11-09).** Predictions: Stage J full 25-pair pipeline run
(`work/stageJ/outputs/stagej_out/`, 25/25 pairs passed all gates). Ground truth: the
Stage E frozen release `dual_camera_gt_v1.0_ai_assisted_audited`.

> **Ground-truth status.** The reference is an **AI-assisted, multi-pass-audited**
> annotation (`AI_ASSISTED_AUDITED`). It is **not human-verified ground truth**, and
> nothing in this report should be read as accuracy against human judgement.

**Protocol.** All matching rules, exclusions, counting bases and the event-matching
rule were fixed in `work/stageK/EVAL_PROTOCOL.md` **before any metric was computed**
(frozen 2026-11-09, sha256
`466f5f316a56536429088f44fe54c05f30161862fda4b8201b616c0e488b9ec2`). **No protocol
amendments were required.** No tuning of any kind was performed: the operating point
is exactly what Stage J ran (helmet_conf 0.25 as configured). Confidence intervals
are pair-level bootstrap percentile 95% CIs (25 pairs resampled with replacement,
B = 2000, seed 20261109; `work/stageK/bootstrap_cis.json`).

## 1. The framing that governs every number below

Stage J **bit-reproduces Pass 1** (44,311 detections, delta 0 in all 200
pair×camera×category cells). The frozen GT was *constructed from* those same
Pass 1 outputs, then corrected by 86 audited correction operations. Every
pipeline-vs-GT divergence therefore decomposes into exactly two sources:

- **(a) Correction delta** — the 86 correction operations themselves (23 excluded
  false-positive tracks, constant-state fixes, validity layer). The pipeline ran on
  *uncorrected* inputs, so these appear as "errors" that are actually the audit
  layer working as intended.
- **(b) GT-construction effects** — GT entities exist only where the annotation
  coupled a detection to a rider; uncoupled detections have no GT counterpart by
  construction, and coupled observations are stored *verbatim* from the detections.

Consequences the reader must keep in mind: observation-level agreement on coupled
units is **perfect by construction** (boxes and states are verbatim copies), so the
informative content of this evaluation lives in (i) the false-positive side and its
decomposition, (ii) aggregation-level behaviour (track-level helmet states,
association, events), and (iii) synchronization. Headline numbers are never quoted
below without their decomposition.

## 2. Headline results

| Family | Metric | Value | 95% CI | Basis |
|---|---|---|---|---|
| Detection — head | Precision (pooled) | 0.689 | [0.594, 0.780] | 6,759 GT head observations |
| Detection — head | Recall (pooled) | 1.000 | [1.000, 1.000] | 6,759 GT head observations |
| Detection — head | F1 (pooled) | 0.816 | [0.745, 0.876] | 6,759 GT head observations |
| Detection — head | AP50 / AP50:95 (pooled) | 0.731 / 0.731 | — | 6,759 GT head observations |
| Detection — motorcycle | Track-level P / R / F1 | 0.820 / 1.000 / 0.901 | — | 82 GT / 100 pred tracks |
| Rider | F1, mean IoU (primary) | 1.000, 1.000 | [1.000, 1.000] | 6,727 spatially-valid coupled obs |
| Helmet — observation | F1 HELMET / NO_HELMET | 1.000 / 1.000 | [1.000, 1.000] | 6,753 decided GT observations |
| Helmet — track | Agreement rate | 0.641 | — | 209 decided GT rider tracks |
| Tracking | IDF1 (head basis) | 0.9987 | [0.997, 0.9998] | 6,759 GT / 6,776 pred coupled units |
| Tracking | ID switches | 0 | — | head-observation basis |
| Association — moto | LINKED P / R / F1 | 0.833 / 1.000 / 0.909 | F1 [0.737, 1.000] | 70 GT / 85 pred units (bases differ, §7) |
| Association — moto | Pred abstention rate | 38.8% | — | 33 / 85 pred units |
| Events | P / R / F1 (strict) | 0.753 / 0.914 / 0.826 | F1 [0.729, 0.927] | 70 GT / 85 pred events |
| Events | Timing error (onset/end) | 0.0 s mean/median | — | 34 windowed strict matches |

## 3. Detection

**Head (box level, IoU ≥ 0.50, operating point as run).** Front: P 0.861 / R 1.000 /
F1 0.925 (TP 5,626, FP 908, FN 0). Back: P 0.346 / R 1.000 / F1 0.514 (TP 1,133,
FP 2,138, FN 0). Recall is 1.0 everywhere by construction (§1b): every GT head
observation is a verbatim stored detection. AP50: front 0.893, back 0.347, pooled
0.731. **AP50 equals AP50:95 in every scope** — a direct fingerprint of the
construction: every matched pair has IoU exactly 1.0 (verbatim copy) and every
unmatched prediction is unmatched at all thresholds, so the PR curve is identical
across the IoU sweep. This is reported, not hidden.

**False-positive decomposition (3,046 FPs, the real signal of this family):**

| FP class | Count | Share | Meaning |
|---|---|---|---|
| Uncoupled detection (no GT entity exists) | 3,024 | 99.3% | GT-construction effect (§1b): detections the annotation never coupled to a rider — concentrated in the back camera (2,138 of its FPs), whose heads frequently belong to riders the coupling stage did not form subtracks for, plus non-rider heads. Not separable into "model error" vs "coupling rejection" from the frozen artifacts. |
| Coupled to a GT-excluded track | 17 | 0.6% | Correction delta (§1a): detections on tracks the audit deleted as false positives. |
| Coupled to a retained track, no GT obs that frame | 5 | 0.2% | Listed individually in `stagek_results.json` (pair_010 back frames 58/60/68/74 on `back_moto_0001_rider_s0`; pair_012 front frame 178 on `front_moto_0004_rider_s0`) — frames where the pipeline holds a coupled observation that the GT stores only as propagated/validity-layer state, not as a direct observation. |

**Motorcycle (track level — the release stores no per-frame GT motorcycle boxes,
so frame-level box metrics and motorcycle AP are NOT COMPUTABLE from the release;
see §9).** All 82 retained GT tracks are present in the predictions with **exact**
first frame, last frame and observed-frame count (mean |Δ| = 0 on all three, 82/82
exact). The 18 predicted tracks with no retained GT counterpart are **exactly the
18 GT-excluded false-positive motorcycle tracks** — i.e., at track level the
pipeline's entire FP count is the correction layer's deletion set, and its FN count
is 0. The pre-registered anchored-box check confirms the construction directly:
**164/164** entry/exit boxes of matched GT tracks appear verbatim in the same
frame's raw Stage J detections.

**Person: NOT COMPUTABLE.** The release contains no GT person entities of any kind.
No person GT was fabricated. Substitutes: Stage J's exact person-count reproduction
(22,350 boxes, delta 0 vs Pass 1) and the head/rider families, which cover the
person-derived evidence the annotation actually used.

## 4. Rider family (Stage C carry-forward honoured)

Primary view = coupled observations whose `rider_frame_validity.csv` row has
`spatial_use_valid = true` (**6,727 units**, the Stage E validity layer; pair_002's
spatially unusable frames and the 4 INVALID rows are excluded by that flag, and 32
GT head observations fall outside the primary view for exactly those reasons).
Result: **P/R/F1 = 1.000, mean IoU = 1.000** — expected under §1b (the GT stores
the pipeline's own coupled boxes verbatim on these rows; the validity layer was
built to identify precisely the rows where that placement is trustworthy, and on
those rows placement is the pipeline's own). The all-coupled sensitivity view (all
6,759 GT observations, no validity filter) is identical. Context counts: 17
predicted coupled observations sit on GT-excluded tracks (correction delta);
0 predicted coupled observations exist on non-inventory retained tracks. The rider
family therefore certifies **self-consistency of the coupling + validity layer**,
not independent localization accuracy — independent placement error was measured
visually in Stages B/D (≈2.4% coupled-stratum defect; 1.6% in the Stage D sweep)
and is not re-estimated here.

## 5. Helmet state

**Observation level** (6,753 decided GT observations; the 6 `occluded_unknown` GT
observations are excluded per protocol and counted): confusion matrix is diagonal —
HELMET 1,365/1,365, NO_HELMET 5,388/5,388; per-class P/R/F1 all 1.000; zero GT
observations lack a predicted counterpart. Again §1b: per-frame GT states on direct
observations are the detections' own states. The SET_CONSTANT correction
decomposition is empty at this level because the constant-state corrections acted
on track/evidence-level state, not on direct per-frame observations — stated
explicitly so the empty decomposition is not misread as the corrections having no
effect; their effect appears at track level:

**Track level** (209 decided GT rider tracks; predicted state = argmax of the
pipeline's `aggregated_confidence`, UNCERTAIN below 0.5): **agreement 134/209 =
64.1%**. Disagreements: **73 tracks where the pipeline's own aggregation is
UNCERTAIN** (55 GT NO_HELMET, 18 GT HELMET — these subtracks carry *zero*
aggregated-confidence mass, `{helmet: 0.0, no_helmet: 0.0}`, i.e. the tracker's
evidence accumulation never fired for them, including all of pair_002's subtracks),
and only **2 hard cross-disagreements** (1 HELMET→NO_HELMET, 1 NO_HELMET→HELMET).
Interpretation: the pipeline's per-frame helmet evidence is exactly what the GT
adopted, but its *track-level evidence aggregation* abstains on a third of rider
tracks — the single largest genuine pipeline-vs-release divergence in this
evaluation, and a property of the aggregation layer, not of detection.

## 6. Tracking (head-observation basis)

IDF1 **0.9987** [0.997, 0.9998]; ID precision 0.9975, ID recall 1.0000; **ID
switches 0**. The entire identity loss is IDFP = 17 — precisely the coupled
observations on GT-excluded tracks (correction delta). **HOTA is NOT computed**:
the release contains no per-frame box trajectories for motorcycle or person
entities, so standard HOTA is not computable from the frozen artifacts; a
head-only variant would be a non-standard number and is not presented as HOTA.
Substitutes reported: IDF1 + ID switches here, motorcycle inventory/temporal
agreement in §3.

## 7. Association

Abstentions (`AMBIGUOUS_ABSTAIN`, `INSUFFICIENT_EVIDENCE`) are reported as their
own rates and are **never** counted as correct or as errors in any P/R denominator.

**Motorcycle (primary).** GT linked pairs 10, predicted linked pairs 12 → TP 10,
FP 2, FN 0: **P 0.833 / R 1.000 / F1 0.909** [0.737, 1.000]. Predicted abstention
rate **38.8%** (33/85: 27 INSUFFICIENT_EVIDENCE + 6 AMBIGUOUS_ABSTAIN). The 2 FP
links are `pair_009 (front_moto_0001, back_moto_0001)` and
`pair_015 (front_moto_0003, back_moto_0001)` — both are correction-layer artefacts:
in each case the corrected GT represents the same physical situation with the
back/front counterpart excluded or split (see the event decomposition, §8, where
the same pairs recur). Outcome cross-tab over the union of decision units (full
table in `stagek_results.json`): on shared units the outcome agreement is exact
for LINKED (10), UNMATCHED (34), INSUFFICIENT (19) and ABSTAIN (4); the remaining
cells are basis differences — the GT's 70 rows are *event-derived* while the
pipeline emits 85 result units (the SB-006 basis difference disclosed at freeze),
so ABSENT cells (GT-only 3, pred-only 18) reflect unit-set construction, not
outcome disagreement.

**Orphan riders (secondary).** 1 GT linked pair, 1 predicted, identical → F1 1.000
(n = 1; no inferential weight). Predicted abstention **63.3%** (76/120). Outcome
agreement on shared units is exact in every category (LINKED 1, UNMATCHED 42,
INSUFFICIENT 53, ABSTAIN 19).

**Rider kind: item-level NOT COMPUTABLE** — Stage J's `association.json` contains
no per-item rider results, only aggregate outcome counts. Aggregate comparison:
GT (17 rows) LINKED 4 / UNMATCHED 4 / INSUFFICIENT 8 / ABSTAIN 1 vs predicted
(aggregate over 25 pairs, 20 units) LINKED 4 / UNMATCHED 6 / INSUFFICIENT 9 /
ABSTAIN 1 — close, on non-identical bases; no item-level claim is made.

## 8. Events

Strict pre-registered matching (protocol §8: same pair + identical constituent
track set + type agreement + temporal IoU ≥ 0.50 where windows exist; GT
null-window events — exactly the 24 INSUFFICIENT + 6 NO_VIOLATION — match on
identity alone): 65 identity matches, 64 type-agreeing → **P 0.753 / R 0.914 /
F1 0.826** [0.729, 0.927]. Per type: DRIVER_NO_HELMET F1 0.852 (23/27/27);
BOTH_NO_HELMET F1 0.842 (8 TP, FP 1, FN 2); PILLION_NO_HELMET F1 0.857 (3 TP,
FP 1); NO_VIOLATION F1 1.000 (6/6); INSUFFICIENT_VISUAL_EVIDENCE F1 0.762
(R 1.000, P 0.615 — all 24 GT cases found, 15 extra predicted).

**Timing errors** on the 34 type-agreeing matches with windows on both sides:
Δonset and Δend are **0.0 s in mean, median and mean-absolute** — the GT windows
were derived from the same coupled histories, so this certifies derivation
consistency on the back-reference timeline (subject to the GAP-1 drift caveat,
§9 of the sync section: unapplied drift adds up to ±76 ms of wall-clock
uncertainty on the drift-affected pairs).

**Decomposition of all 27 strict error instances (21 FP + 6 FN) — residual
unexplained = 0:**

| Class | FP | FN | Content |
|---|---|---|---|
| Correction delta — event involves a GT-excluded track | 18 | — | Events the pipeline derives on tracks the audit deleted (15 INSUFFICIENT_VISUAL_EVIDENCE, 2 DRIVER_NO_HELMET, 1 BOTH_NO_HELMET). |
| Would-match — track sets differ only by excluded track(s) | 2 | 5 | The pair_020 pattern and its kin (pair_009, pair_012, pair_015): the corrected GT splits or drops the excluded side, the pipeline links it. Same physical events, different constituent sets *because of* the correction layer. |
| Type mismatch on identical track set | 1 | 1 | pair_002: GT `BOTH_NO_HELMET` vs predicted `PILLION_NO_HELMET` on `front_moto_0001` alone, temporal IoU 1.0 — a genuine rider-role attribution disagreement, in the pair already flagged spatially unusable (FLAG_PAIR_SPATIALLY_UNUSABLE). |

Read together: **on retained tracks, with constituent sets compared after the
correction layer is accounted for, the pipeline's event derivation disagrees with
the frozen release in exactly one place — the pair_002 role attribution.** The
strict F1 of 0.826 is the honest pre-registered number; the decomposition is what
makes it interpretable, and neither number is quoted without the other.

## 9. Synchronization summary

From Stage J `sync_diagnostics.json` (full 25-row table: `work/stageK/sync_by_pair.csv`;
decoded frame counts per camera included there — Stage J gates verified decoded ==
Phase 12C forensic counts for all 50 videos: front 6,557 / back 8,684).

| Stratum | Pairs | Median ε_sync | Median |drift| | Max |drift| | Median Q5 skew max |
|---|---|---|---|---|---|---|
| Pooled | 25 | 3 ms | 0 ms | 76 ms | 7.8 ms |
| Near-frame (004/005/008/015) | 4 | 36 ms | 72 ms | 76 ms | 5.4 ms |
| MEDIUM quality (008/019/022) | 3 | 8 ms | 16 ms | 76 ms | 3.5 ms |

**GAP-1 caveat (mandatory):** clock drift is *reported, never applied* anywhere in
the system, so where drift ≠ 0 the stored ε_sync is a **lower bound** on
end-of-window misalignment. Per Stage F design OQ-2, the labelled bound
ε_sync + |drift| is reported alongside (never instead of) the raw values for pairs
004/005/008/015/016/019 in `sync_by_pair.csv` (e.g. pair_008: ε 38 ms, drift
−76 ms → bound 114 ms). Q5 nearest-available-frame errors stay in the
single-digit-ms range in median even on the near-frame pairs; the dominant
synchronization uncertainty in this system is unapplied drift, not frame lookup.

## 10. What was NOT computed, and why

- **Person detection P/R/F1/AP** — no GT person entities exist in the release.
- **Motorcycle frame-level box P/R and AP** — the release stores track-level
  motorcycle records only (entry/peak/exit boxes), no per-frame GT boxes.
- **HOTA** — requires per-frame trajectories for the tracked entity classes;
  unavailable in the release for motorcycle/person (see §6).
- **Rider association, item level** — Stage J emits no per-item rider results
  (aggregate counts only; §7).
- No threshold sweeps, no confidence re-thresholding, no post-hoc rule changes:
  the operating point is Stage J's, exactly as run.

## 11. Limitations

1. **Circularity bound (fundamental).** The GT was constructed from the same
   detector outputs the pipeline reproduces; observation-level families therefore
   measure construction/correction consistency, not detector accuracy against
   independent human judgement. The genuinely independent error evidence in this
   programme remains the visual audits (Stages B/D, Passes 2/3) and the
   aggregation-level divergences surfaced here (helmet track-level abstention,
   association FP links, pair_002 role attribution).
2. The head-detection FP class "uncoupled" (3,024) cannot be split into detector
   false alarms vs coupling rejections from the frozen artifacts; the back
   camera's low box-level precision (0.346) should be read with that caveat and
   with Phase 12D's known rear-view detection behaviour.
3. Bootstrap CIs for the by-construction-perfect metrics are degenerate [1, 1];
   they describe pair-resampling stability only.
4. Event timing errors are on the back-reference timeline with drift unapplied
   (GAP-1); wall-clock timing carries the additional bounded uncertainty of §9.
5. Orphan association F1 rests on a single linked pair; rider association is
   aggregate-only. Both are reported for completeness, not as evidence.
6. pair_002 is spatially unusable (Stage C flag); its event type mismatch and
   zero-confidence subtracks should be read in that light.

## 12. Artifacts (all in `work/stageK/`)

`EVAL_PROTOCOL.md` (frozen pre-registration) · `STAGE_K_PROGRESS.md` ·
`stagek_results.json` (all families, full detail) · `metrics_summary.csv` ·
`per_pair_stats.json` (bootstrap inputs) · `bootstrap_cis.json` ·
`sync_by_pair.csv` · `events_detail.csv` · `extract.py` / `evaluate.py` /
`bootstrap.py` · `evaluate.log` / `bootstrap.log` / recon logs.
Nothing outside `work/stageK/` and this report was created or modified by Stage K
evaluation code; the pipeline, weights, configs, release, package, Stage J outputs
and tests are untouched.
