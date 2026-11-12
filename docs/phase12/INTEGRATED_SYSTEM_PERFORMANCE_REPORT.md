# Integrated Dual-Camera Helmet System — PERFORMANCE REPORT (Stage L)

**Status: COMPLETE (2026-11-09).** Runtime/memory figures are taken from the
Stage I and Stage J run artifacts; all quality metrics are taken from the Stage K
artifacts in `work/stageK/` (`metrics_summary.csv`, `stagek_results.json`,
`bootstrap_cis.json`), computed under a protocol frozen before any metric existed
(`EVAL_PROTOCOL.md`, sha256 `466f5f316a56536429088f44fe54c05f30161862fda4b8201b616c0e488b9ec2`,
zero amendments, no tuning — the operating point is exactly what Stage J ran).

> **Read the framing first.** The reference is the frozen release
> `dual_camera_gt_v1.0_ai_assisted_audited` — **AI-assisted, not human-verified**.
> Stage J bit-reproduces Pass 1 and the GT was constructed from Pass 1 + 86
> corrections, so observation-level agreement on coupled units is perfect *by
> construction* (boxes and states stored verbatim). The informative content of
> this evaluation is the false-positive side and its decomposition, the
> aggregation-level divergences (track-level helmet states, association, events),
> and synchronization. Every headline below is quoted with its decomposition.

## 1. Runtime and memory

| Run | Venue | Wall time | Peak memory | Notes |
|---|---|---|---|---|
| Stage I verification run (pair_001, 603 frames) | VM CPU | **160.7 s** (0.267 s/frame, both models) | **≈981 MB RSS** | COMPLETE, summary validation 8/8; detector calls == decoded frames (206 front / 397 back) — new-frame-only with real models; detection totals delta 0 vs Pass 1 in every category |
| Stage J full run (25 pairs, 15,241 frames) | Kaggle T4 (`elafahmed6563/helmet-stagej-pipeline-full-run` v1) | **kernel wall 958 s**; sum of per-pair walls 675.7 s; max single-pair wall 52.7 s | **max per-pair peak RSS 2,164.5 MB** (min 2,142.4 MB across pairs) | One worker subprocess per pair via the pipeline CLI, `device=cuda`; 25/25 pairs COMPLETE, all gates passed |
| Pass 1 annotation run (reference) | Kaggle T4 | 644 s camera wall time (~11 min kernel) | — | The corpus the pipeline reproduces |

**Kaggle usage:** pre-flight quota read 25.28 GPU-hours remaining of 30.00 h
(used 4.72 h, refresh 2026-11-14); post-Stage-J remaining **25.01 h**. Stage J
consumed **≈0.27 GPU-hours**; it was never queued or rejected. The pipeline's
memory profile (≤ ~2.2 GB peak RSS per pair on GPU; ~1 GB on CPU) sits well
inside the Stage F design budget for the Apple M1 8 GB reference machine.

## 2. Detection

**Head (box level, IoU ≥ 0.50, basis 6,759 GT direct coupled observations):**

| Scope | P | R | F1 | AP50 | TP / FP / FN |
|---|---|---|---|---|---|
| Pooled | **0.689** [0.594, 0.780] | 1.000 | **0.816** [0.745, 0.876] | **0.731** | 6,759 / 3,046 / 0 |
| Front | 0.861 | 1.000 | 0.925 | 0.893 | 5,626 / 908 / 0 |
| Back | 0.346 | 1.000 | 0.514 | 0.347 | 1,133 / 2,138 / 0 |

CIs are pair-level bootstrap percentile 95% (B = 2000, seed 20261109). Recall is
1.0 by construction (every GT observation is a verbatim stored detection), and
AP50 == AP50:95 in every scope because every matched pair has IoU exactly 1.0 —
both are fingerprints of the construction, reported openly in the Stage K report.

**False-positive decomposition (3,046 — the real signal of this family):**
3,024 uncoupled detections (99.3%; no GT entity exists for them — a
GT-construction effect, concentrated in the back camera whose riders the coupling
stage often did not form subtracks for; not separable into model error vs
coupling rejection from the frozen artifacts), 17 on correction-excluded tracks
(correction delta), 5 individually listed coupled cases.

**Motorcycle (track level — the release stores no per-frame GT motorcycle boxes):**
P 0.820 / R 1.000 / **F1 0.901** (82 GT tracks, 100 predicted). All 82 GT tracks
match exactly on first frame, last frame and hit count; **all 18 FP tracks are
exactly the correction-excluded set**; the anchored-box check confirms 164/164
entry/exit boxes verbatim in the same frames' raw detections.

**Person: NOT COMPUTABLE** — the release contains no GT person entities; nothing
was fabricated. Substitute evidence: Stage J reproduced the Pass 1 person count
exactly (22,350 boxes, delta 0).

## 3. Rider localization (spatially-valid primary view)

Primary basis = the 6,727 coupled observations whose validity-layer row has
`spatial_use_valid = true` (Stage C layer; pair_002's spatially unusable rows and
the 4 INVALID rows excluded by that flag): **P / R / F1 = 1.000, mean IoU 1.000**
— a construction self-consistency result (the GT stores the pipeline's own
coupled boxes verbatim on these rows), *not* independent localization accuracy.
The independent placement evidence is visual: Stage B's fresh coupled-stratum
sample measured a 2.4% placement defect rate; Stage D's sweep measured 1.6%.
Context: 17 predicted coupled observations sit on GT-excluded tracks (correction
delta); 0 exist on non-inventory retained tracks.

## 4. Helmet state

**Observation level** (6,753 decided GT observations; 6 `occluded_unknown`
excluded per protocol): confusion matrix is diagonal — HELMET 1,365/1,365,
NO_HELMET 5,388/5,388; per-class P/R/F1 all **1.000** (again by construction on
direct observations).

**Track level** (209 decided GT rider tracks; predicted state = argmax of the
tracker's aggregated confidence, UNCERTAIN below 0.5): **agreement 134/209 =
64.1%** — the largest genuine divergence in the evaluation. 73 tracks are
pipeline-UNCERTAIN (55 GT NO_HELMET, 18 GT HELMET): these subtracks carry *zero*
aggregated-confidence mass — the tracker's evidence accumulation never fired for
them, including all of pair_002's subtracks. Only **2 hard cross-disagreements**
(1 HELMET→NO_HELMET, 1 NO_HELMET→HELMET). The defect is in the aggregation layer,
not in per-frame detection.

## 5. Tracking (head-observation basis)

**IDF1 0.9987** [0.997, 0.9998]; ID precision 0.9975, ID recall 1.0000;
**ID switches 0**; IDTP 6,759 / IDFP 17 / IDFN 0 — the entire identity loss is
the 17 coupled observations on correction-excluded tracks. **HOTA NOT
COMPUTED**: the release lacks per-frame box trajectories for motorcycle/person,
and a head-only variant would be a non-standard number; declared, not fabricated.

## 6. Cross-camera association

Abstentions (`AMBIGUOUS_ABSTAIN`, `INSUFFICIENT_EVIDENCE`) are reported as their
own rates and are **never** counted as correct or as errors in any denominator.

| Family | P | R | F1 | Abstention |
|---|---|---|---|---|
| Motorcycle (70 GT / 85 pred decision units) | 0.833 | 1.000 | **0.909** [0.737, 1.000] | **38.8%** (33/85) |
| Orphan riders (115 GT / 120 pred units) | 1.000 | 1.000 | 1.000 (n_linked = 1; no inferential weight) | **63.3%** (76/120) |

The 2 FP motorcycle links — pair_009 (front_moto_0001 × back_moto_0001) and
pair_015 (front_moto_0003 × back_moto_0001) — are both correction-layer
artefacts (the corrected GT represents the same physical situations with one
side excluded or split). **Rider item-level association: NOT COMPUTABLE** —
Stage J emits no per-item rider results; aggregate counts are close (GT LINKED 4 /
UNMATCHED 4 / INSUFFICIENT 8 / ABSTAIN 1 vs predicted 4 / 6 / 9 / 1) on
non-identical bases, and no item-level claim is made.

## 7. Events (strict pre-registered matching)

Same pair + identical constituent track set + type agreement + temporal IoU ≥
0.50 where windows exist (GT null-window events — the 24 INSUFFICIENT + 6
NO_VIOLATION — match on identity alone):

| | P | R | F1 |
|---|---|---|---|
| Overall (70 GT / 85 pred) | 0.753 | 0.914 | **0.826** [0.729, 0.927] |
| DRIVER_NO_HELMET | 0.852 | 0.852 | 0.852 (TP 23, FP 4, FN 4) |
| BOTH_NO_HELMET | 0.889 | 0.800 | 0.842 (TP 8, FP 1, FN 2) |
| PILLION_NO_HELMET | 0.750 | 1.000 | 0.857 (TP 3, FP 1, FN 0) |
| NO_VIOLATION | 1.000 | 1.000 | 1.000 (TP 6) |
| INSUFFICIENT_VISUAL_EVIDENCE | 0.615 | 1.000 | 0.762 (TP 24, FP 15, FN 0) |

**Timing errors** on the 34 type-agreeing windowed matches: Δonset and Δend are
**0.0 s in mean, median and mean-absolute** (back-reference timeline; the GT
windows derive from the same coupled histories, so this certifies derivation
consistency — wall-clock timing additionally carries the GAP-1 drift bound, see
the Synchronization Report).

**Decomposition of all 27 error instances (21 FP + 6 FN) — residual unexplained
= 0:** 18 FP are correction-delta (events on GT-excluded tracks); 2 FP + 5 FN
are would-match cases whose track sets differ only by excluded tracks (pairs
009/012/015/020); 1 type mismatch on an identical track set (pair_002: GT
BOTH_NO_HELMET vs predicted PILLION_NO_HELMET, temporal IoU 1.0 — a genuine
rider-role attribution disagreement in the pair already flagged spatially
unusable). Read together: after the correction layer is accounted for, the
pipeline's event derivation disagrees with the frozen release in exactly one
place — the pair_002 role attribution. The strict F1 of 0.826 is the honest
pre-registered number; neither number is quoted without the other.

## 8. What was not computed

Person detection P/R/F1/AP (no GT person entities) · motorcycle frame-level box
P/R and AP (no per-frame GT boxes in the release) · HOTA (no per-frame
trajectories for the tracked classes) · rider association at item level (no
per-item predictions). Each was declared NOT COMPUTABLE in the frozen protocol
with a pre-registered substitute; none was fabricated. Full detail:
`repo/docs/phase12/POST_ANNOTATION_STAGE_K_EVALUATION.md` and `work/stageK/`.
