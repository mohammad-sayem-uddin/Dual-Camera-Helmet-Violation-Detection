# TASK 3 — Frame-Count Reconciliation: 15,637 (Phase 12C prose) vs 15,241 (Pass 1 / Stage J)

**Date:** 2026-11-12 · **Scope:** discrepancy of 396 frames · **Data:** `frame_reconciliation.csv`
(50 rows; copy in `repo/results/phase12_postcompletion/frame_count_reconciliation.csv`),
raw measurements in `work/postcompletion/frame_counts_measured.csv`.
No count was altered to force agreement; no frozen artifact was modified.

---

## 1. Verdict

**The 396-frame difference is a prose-only error in the two Phase 12C report texts. There is no
frame-count discrepancy in any data artifact, and no processing defect.**

- The figure **15,637 appears only as a prose total** in exactly two places:
  - `repo/docs/phase12/PHASE12C_FORENSIC_TIMESTAMP_VALIDATION_REPORT.md`, line 15:
    "Across all 50 video containers (`25 front + 25 back`, totaling `15,637` frames)"
  - `repo/docs/phase12/PHASE12C_PTS_SYNC_IMPLEMENTATION_REPORT.md`, line 35:
    "Across all 50 videos (15,637 total frames)"
- **Phase 12C's own machine-generated forensic artifact contradicts the prose.** The audit's output
  `repo/results/phase12c_pts_sync/stream_pts_forensics_50_videos.csv` (50 rows) sums to exactly
  **15,241** (front 6,557 + back 8,684), and its `frame_count` equals its
  `opencv_decoded_frame_count` on **every one of the 50 rows** (`frame_count_exact_match = True`
  throughout; `phase12c_pts_validation_summary.json`: `all_50_frame_counts_exact_match = true`).
  The forensic report's own §1.2 per-stream table (e.g. pair_001_front 206, pair_013_back 608,
  pair_014_back 276, pair_015_back 190, pair_016_back 310, pair_017_back 200, pair_018_back 198)
  is likewise consistent with the 15,241 accounting, not with 15,637.
- The companion Phase 12C-derived figure used everywhere downstream — e.g.
  `repo/docs/phase12/PHASE12_ANNOTATION_PRE_EXECUTION_INVENTORY.md` line 15, quoting the
  "Phase 12C forensic audit" — is **15,241 = 6,557 front + 8,684 back**.

**Composition of the 396:** 15,637 − 15,241 = 396 is composed **entirely of a prose aggregate
error: 396 frames that exist in no count of any kind.** It cannot be decomposed per video or per
cause, because **every video's contribution to the difference is 0** (see §3): all 50 videos agree
exactly across the Phase 12C artifact counts, four independent fresh measurements, and the
Pass 1 / Stage J processed and coverage counts.

## 2. What each number actually represents

| Figure | What it is | Source |
|---|---|---|
| 12C forensic CSV `frame_count` | Video-stream **packet count** = length of the ffprobe packet-PTS list (`FramePTSIndex.frame_count = len(self.pts_ticks)`, `repo/src/synchronization/pts_index.py:73`) | `results/phase12c_pts_sync/stream_pts_forensics_50_videos.csv` |
| 12C forensic CSV `opencv_decoded_frame_count` | **Sequential decode count** via `cap.grab()` (`validate_pts_against_opencv_sequential`, pts_index.py) | same CSV |
| **15,241** | Sum of both of the above (identical per video): front 6,557 + back 8,684 | same CSV; inventory doc line 15 |
| **15,637** | **A prose total only.** No artifact, CSV, JSON, or metadata in the repo contains per-video numbers summing to it; its origin is not recoverable from the phase's artifacts | forensic report line 15; implementation report line 35 |
| Pass 1 / Stage J counts | Sequential decode of the **full streams** (not overlap-restricted); `frames_decoded_sequential` / `frames_decoded`, each asserted equal to `frames_expected_container` with `coverage_complete = true` | `work/pass1/pair_*/pair_*_summary.json`; `work/stageJ/outputs/stagej_out/pair_*/pair_summary.json`; `work/stageJ/validation_results.json` |

Candidate recombinations of the phase's own pair-level table also fail to reproduce 15,637:
60 Hz reference ticks total 5,686; unique frames in overlap total 10,636; suppressed duplicate
ticks total 736 (15,241 + 736 = 15,977); ticks + unique = 16,322. None equals 15,637.

## 3. Per-video reconciliation (summary)

Full table: `frame_reconciliation.csv` (50 rows × 7 count columns + excluded/difference).
Result: **all 50 rows are in perfect agreement** — for every video,

> Phase 12C artifact decoded = Phase 12C artifact packet = measured packet count =
> measured `nb_frames` = measured sequential decode = Stage J processed = Stage J coverage,
> **excluded_frames = 0, difference = 0.**

- Total of every count column: **15,241**. Videos that differ: **none (∅)**.
- Front totals: 6,557 by every method. Back totals: 8,684 by every method.
- Fresh measurement (2026-11-12, VM copies — SHA-256 identical to the Mac originals per the
  Phase 11 manifest and the Stage L forensic audit): ffprobe `-count_packets` video packet count,
  ffprobe `nb_frames` metadata, OpenCV `CAP_PROP_FRAME_COUNT`, and OpenCV sequential `grab()`
  decode count — **four conventions, one answer: 15,241**, agreeing per video on all 50 files
  (`work/postcompletion/frame_counts_measured.csv`).

## 4. Candidate causes — tested against evidence

| Candidate cause | Verdict | Evidence |
|---|---|---|
| Different decoding conventions (packet vs decoded) | **REFUTED** | Packet count == sequential decode count on all 50 streams — in the 12C forensic CSV itself and in today's independent measurement. |
| `nb_frames` metadata inaccuracy | **REFUTED** | Σ `nb_frames` = 15,241; equals packet count per video on all 50 files. |
| Excluded invalid frames | **REFUTED** | excluded = 0 on all 50 rows; decode == packet everywhere means no undecodable packets; no exclusion record exists in Pass 1 / Stage J outputs. |
| Common-overlap restriction | **REFUTED** | Stage J/Pass 1 processed **full streams**: e.g. pair_001 back processed 397 frames vs 187 unique-in-overlap; pair_013 back 608 vs 201. Overlap-restricted processing would total 10,636, not 15,241. |
| Different source inventory version | **REFUTED** | Same 50 files; VM copies SHA-256-verified identical to the Mac originals audited by Phase 11/12C (Stage L forensic audit: 50/50). |
| Missing frame coverage / processing defect | **REFUTED** | `coverage_complete = true` in all 50 Pass 1 and all 25 Stage J pair summaries; decoded == container-expected on every camera of every pair; coverage CSVs carry one row per decoded frame (spot-verified pair_001: 206/397). |
| Erroneous prose aggregate in the 12C reports | **CONFIRMED as the sole cause** | 15,637 exists only in report prose (2 locations); the phase's own generated data sums to 15,241; no recombination of the phase's tables reproduces 15,637. |

## 5. Explicit answer — did any decodable source frame go unprocessed?

**NO.** Every frame that exists in the 50 source videos — 15,241 frames by packet count,
`nb_frames` metadata, `CAP_PROP_FRAME_COUNT`, and sequential decoding alike — was decoded,
processed, and covered by both Pass 1 and Stage J, with per-pair, per-camera coverage verified
complete and decoded counts equal to container counts throughout. The set of unprocessed
decodable source frames is **empty**. The 396 "missing" frames never existed in the videos;
they exist only in a sentence in two Phase 12C reports.

## 6. Recommended correction (not applied — reports are frozen records)

The two prose occurrences of "15,637" should read "15,241" if the Phase 12C reports are ever
re-issued. Per the post-completion rules, the original reports were **not** edited by this task;
this document is the reconciliation record.
