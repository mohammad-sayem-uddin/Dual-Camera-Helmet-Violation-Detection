# Post-Completion Reproducibility and Validation Report

**Project:** Dual-Camera Context-Aware Helmet Violation Detection
**Repository:** `mohammad-sayem-uddin/Dual-Camera-Helmet-Violation-Detection`, branch `main`
**Date:** 2026-11-12
**Scope:** Final delivery, reproducibility and scientific validation audit ordered after the
Stage L completion claim (16/16 mission criteria). This report covers Tasks 1–8 of that order.
Per the order's stop rule, no OCR, model training, fine-tuning or new implementation phase was
begun; work stops at this report.

---

## 1. Task 1 — Transfer of deliverables to the Mac repository

**Target:** `/Users/sayemuddin/Desktop/research_helmet` (the repository of record).

A single verified sync package was built on the VM and transferred intact:

- Package: `helmet_mac_sync_postcompletion_2026-11-12.zip`, **22,521,615 bytes**, **530 files**
  (529 deliverables + `PACKAGE_MANIFEST.sha256` listing the SHA-256 of the other 529).
- `unzip -t` clean; all **136 files** that already existed in the VM working mirror were verified
  byte-identical to their sources before packaging.
- The zip was written directly into the Mac repository root through the paired-device file
  service (bytes written = 22,521,615, exact match) and unzipped by the owner in Terminal.

**Contents transferred:** integrated pipeline implementation (`src/pipeline/`), the frozen
annotation release `dual_camera_gt_v1.0_ai_assisted_audited` (44 files), the annotation working
package (40 files), annotation audit and correction reports, synchronization engine design and
reports (Stages A–L, `docs/phase12/`), the full 25-pair Stage J inference results
(`results/phase12_stagej/`), integrated evaluation results (`results/phase12_integrated/`, 9
files), post-completion audit outputs (`results/phase12_postcompletion/`), the Stage H pipeline
tests (3 files), reproducibility scripts (`scripts/reproducibility/`, 10 scripts), checksums for
every package, and `SYNC_INSTRUCTIONS_V2.md`.

**Deliberately not transferred:** temporary caches, credentials, duplicate source videos and
model weights (already present on the Mac and git-ignored), and the old `_chunks/` transfer
leftovers. Original Git history and existing user changes were preserved; nothing on the Mac was
overwritten outside the deliverable paths.

## 2. Task 2 — Verification of the transfer

Verified on the Mac from the owner's Terminal output (2026-11-12) and by direct reads of the
Mac working tree:

| Check | Result |
|---|---|
| Release checksums (`releases/…/checksums/release_checksums.sha256`) | **43 / 43 OK** |
| Annotation package checksums | **39 / 39 OK** |
| Integrated results checksums (`results/phase12_integrated/`) | **8 / 8 OK** |
| Stage J pair result directories | **25 / 25 present** |
| Annotation release files | **44 / 44 present** |
| Integrated pipeline present (`src/pipeline/`) | Yes |
| `docs/phase12/` reports incl. the 3 new post-completion documents + 18 evidence PNGs | All present (39 entries) |
| Pre-commit `git status` | Only intended deliverables untracked; no videos, no credentials staged |

A successful copy was **not** assumed to imply validity: every package carries and passed its own
SHA-256 manifest on the destination machine.

## 3. Task 3 — Frame-count reconciliation (15,637 vs 15,241)

**Verdict: the Phase 12C figure of 15,637 is a prose-only error.** Full analysis:
`docs/phase12/PHASE12_FRAME_COUNT_RECONCILIATION.md`; per-video data:
`results/phase12_postcompletion/frame_count_reconciliation.csv` (50 rows).

- Phase 12C's own machine-generated forensic CSV
  (`results/phase12c_pts_sync/stream_pts_forensics_50_videos.csv`) sums to exactly **15,241**.
  The wrong number appears only as prose in two Phase 12C reports
  (`PHASE12C_FORENSIC_TIMESTAMP_VALIDATION_REPORT.md` line 15;
  `PHASE12C_PTS_SYNC_IMPLEMENTATION_REPORT.md` line 35). Those reports are frozen records and
  were **not** edited; this reconciliation is the correction of record.
- Independent re-measurement on 2026-11-12 under **four conventions** — ffprobe packet count,
  ffprobe `nb_frames`, OpenCV `CAP_PROP_FRAME_COUNT`, OpenCV sequential `grab()` decode — all
  total exactly **15,241** and agree per-video on **50/50** files. Excluded frames = **0** on
  every row.
- **No decodable source frame went unprocessed by Pass 1 or by Stage J; the unprocessed set is
  empty.** The 396-frame difference is not a decoding convention, not excluded invalid frames,
  not a common-overlap restriction, not an inventory change, and not a processing defect.

## 4. Task 4 — Helmet track-level failure analysis (64.1% agreement)

**Full report:** `docs/phase12/HELMET_TRACK_LEVEL_FAILURE_ANALYSIS.md`; per-track data:
`helmet_track_breakdown.csv` (209 decided rider tracks: AGREE 134, PIPELINE_UNCERTAIN 73,
HARD_DISAGREE_FALSE_HELMET 1, HARD_DISAGREE_FALSE_NO_HELMET 1 — reproducing Stage K's confusion
exactly); 18 evidence PNGs in `docs/phase12/track_failure_examples/`; 8 worked examples.

**Headline finding: the 64.1% is an aggregation/export artifact, not a classification failure.**
A track's exported helmet state is read from a 10-frame sliding-window EMA at end-of-run, while
unmatched/abstained frames keep appending undetected observations. Any motorcycle subtrack whose
last coupled observation ends ≥10 stepped frames before its parent track ends therefore exports
zero mass and is reported UNCERTAIN. The separation is perfect in the data: all 73 abstentions
have a tail gap ≥ 10 frames (median 66, max 285); every agreed track has a tail gap < 10.

- **Where the pipeline actually decided, track agreement is 134/136 = 98.5%.**
- In 72 of the 73 abstentions, the pipeline's own observation majority already matched the
  ground truth (65 of them unanimous). Worst case: a pair_024 pillion with 84/84 unanimous
  no-helmet observations discarded over a 226-frame tail gap.
- The 2 hard disagreements are both pair_004 front orphan tracks: `front_orphan_0003` is a
  genuine mid-track helmet-donning transition (pipeline HELMET vs GT NO_HELMET; the GT is
  explicitly not endorsed for the donning window), and `front_orphan_0005` is a sparse detector
  error (pipeline NO_HELMET vs GT HELMET, GT resting on a Stage C visually verified set-constant).
- Driver/pillion identity confusion: **none** (0/23 multi-rider parents). Track fragmentation in
  the scored data: **none**.
- The frozen annotation release was **not** changed. The identified remedy (export the track
  state from the last coupled observation, or report the observation majority alongside the EMA
  state) is a candidate change for a future phase; it was not applied under this order.

## 5. Task 5 — Independent human validation plan (designed, NOT executed)

**Protocol:** `docs/phase12/HUMAN_VALIDATION_PROTOCOL.md`. Design only — **no human review has
been performed and no human-verified labels exist.**

- Census of all 73 pipeline-uncertain tracks, both hard disagreements, all 46 decided events'
  roles, all 70 events re-marked for onset/end, all LINKED and AMBIGUOUS cross-camera links, and
  all 30 low-evidence subtracks; plus a powered random sample of 60 of the 134 agreeing tracks.
- ≈399 decision units, ≈55 person-hours total (two blinded independent reviewers ≈22 h each,
  ≈7 h adjudication, ≈4 h coordination), a 40-item calibration set with a κ ≥ 0.70 entry gate,
  pre-registered pass thresholds, and a sample manifest hash-frozen before review and held out
  from all tuning.
- Human labels, if the protocol is ever executed, must land as a **separate provenance layer**
  and any correction would be issued as a new v1.1 release — the frozen v1.0 release is never
  edited in place. A completion record with reviewer identities, dates, manifest hash and
  measured metrics is required before any human-verification claim may be made.

## 6. Task 6 — Reproducibility check in the Mac environment

**Verified on the Mac (2026-11-12):**

- Phase 9 helmet model `best.pt` SHA-256 =
  `0929a882667e0f7e2b631071ab218a2415cc1b816cdede7fb6812543d840b0e5` — exact match.
- `yolov8n.pt` SHA-256 =
  `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` — exact match.
- Frozen release, annotation package and integrated-results checksums: 43/43, 39/39, 8/8 (§2).
- Original videos and Phase 11 synchronization metadata unchanged (Stage L forensic audit:
  50/50 video SHA-256 vs manifest, 6/6 metadata files byte-identical to the pristine tarball;
  the transfer and commit touched no video or metadata path).

**Test execution on the Mac — environment limitation, recorded separately from defects:**
the Mac's default Homebrew Python 3.14 has no pytest installed
(`/opt/homebrew/opt/python@3.14/bin/python3.14: No module named pytest`) and no project virtual
environment existed. The owner elected on 2026-11-12 to waive building a Mac environment rather
than install one. **The pipeline test suites were therefore not executed on the Mac, and no Mac
test pass is claimed.** The standing test record remains the VM result: the three pipeline suites
pass 54/54 (Stage H), and the full repository suite stands at 140 passed / 3 failed, the 3
failures being pre-existing environment issues identical before and after this work (missing
`soundfile`, missing `easyocr`, and an Apple-Silicon MPS test running on Linux). No test failure
attributable to the delivered pipeline code was found on any machine.

## 7. Task 7 — Git commit and push

- **Commit:** `1a1b3b26c6c7d09d1fccf5daefaf678a37be813e` on `main`, parent
  `6aba04434af80cf0263de031a3c06488bebec29f`, message *"Helmet: annotation release
  dual_camera_gt_v1.0_ai_assisted_audited + pipeline, tests, Stage A-L reports, Stage J outputs,
  post-completion audits"*. **530 files changed, 638,965 insertions(+), purely additive** (every
  file in create mode). The sync zip itself and `_chunks/` were deliberately left uncommitted.
- **Push history, recorded honestly:** the first `git push` failed with
  `fatal: The current branch main has no upstream branch` — standard git behaviour for a branch
  with no configured upstream; nothing was transmitted and no history was affected. The owner
  then ran `git push --set-upstream origin main`, which succeeded.
- **Push verified independently:** GitHub's `main` HEAD was read back through the GitHub API on
  2026-11-12 and equals `1a1b3b26c6c7d09d1fccf5daefaf678a37be813e`. No force-push was used or
  attempted at any point; history is intact and strictly extended.
- This report is delivered in a small follow-up commit on top of `1a1b3b2…`; a document cannot
  contain its own commit's SHA, so that SHA is reported in the delivery message instead.

## 8. Remaining scientific limitations (unchanged by this audit)

1. **The ground-truth release is `AI_ASSISTED_AUDITED`, not human-verified.** It originates from
   the pipeline's own Pass 1 detections plus documented corrections. The project's Phase 12B
   protocol tension (it prohibits unverified model-bootstrapped ground truth) remains disclosed,
   not resolved. The Task 5 protocol is the designed remedy and is unexecuted.
2. **Construction self-consistency:** Stage J reproduces Pass 1 bit-exactly and the GT derives
   from Pass 1 + corrections, so coupled observation-level agreement is perfect by construction.
   The independent evidence for correctness is the visual audit samples (Pass 3, Stages B and D),
   not the coupled metrics.
3. **Track-state export artifact (Task 4):** the EMA end-of-run read discards decided evidence on
   subtracks with tail gaps ≥ 10 frames, producing a 35% track-level abstention rate and the
   misleading 64.1% headline figure. Unfixed by design under this order.
4. **Abstention behaviour:** cross-camera association abstains on 38.8% of rider observations
   (63.3% for orphans); event precision 0.753 / recall 0.914 (Stage K, frozen protocol).
5. **Metric families declared NOT COMPUTABLE in Stage K remain so:** person detection metrics,
   frame-level motorcycle AP, HOTA, item-level rider association. They were never fabricated.
6. **License plates:** visibility-only by protocol; 0 plates were confidently readable, all
   remain `NOT_ASSESSED`. Nothing was invented.
7. **Synchronization:** clock drift (up to ±76 ms on pairs 004/005/008/015) is reported, never
   compensated, so near-frame residuals are lower bounds; no physical genlock or exposure-level
   synchronization was ever established.
8. **Mac-side test execution gap (§6):** reproducibility of the test suites on macOS specifically
   rests on the VM record plus checksum-identical code, weights and data — not on a Mac test run.

---

*End of post-completion audit. Per the ordering instruction, work stops here: no OCR, no model
training, no fine-tuning, and no further implementation phase has been started.*
