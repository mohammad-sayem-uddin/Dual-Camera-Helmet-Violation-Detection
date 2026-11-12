# MASTER REPORT — Dual-Camera Helmet Violation Detection Project

**Prepared:** 2026-11-12 (final handover from Elsa to the project owner)
**Repository of record:** `/Users/sayemuddin/Desktop/research_helmet`
**GitHub:** `mohammad-sayem-uddin/Dual-Camera-Helmet-Violation-Detection`, branch `main`
**Status at handover:** Annotation mission complete · Post-annotation Stages A–L complete (16/16
acceptance criteria) · Post-completion audit Tasks 1–8 complete · Owner takes over from here.

**Owner's statement, recorded 2026-11-12:** the owner states that he has personally verified all
annotations and considers the ground truth complete, and that he will continue the project
himself from this point. This statement is recorded as the owner's. For the scientific record:
the frozen release in the repository remains labelled `AI_ASSISTED_AUDITED` (see §7), and the
human validation protocol (§6.3) has no execution/completion record in the project files. If the
owner wishes the repository itself to carry a human-verified label, that requires issuing a new
v1.1 release — a decision that is now entirely the owner's.

---

# PART A — WHAT HAS BEEN DONE

## 1. Annotation mission (completed 2026-11-09, acceptance gate 8/8)

**Scope executed exactly as ordered:** all 25 synchronized dual-camera pairs, 50 videos, every
decoded frame; sequential decoding only, original PTS preserved (no frame-index seeking on VFR
video); original videos never modified, recompressed, renamed or deleted; Phase 11 sync
manifests/metadata never altered; no model training or fine-tuning; no OCR; no Phase 12F/12G.

| Quantity | Verified value |
|---|---|
| Pairs / videos | 25 / 50 |
| Decoded frames | **15,241** (front 6,557 + back 8,684) |
| Raw detection boxes (Pass 1) | **44,311** — motorcycle 12,156 · person 22,350 · head 9,805 |
| Correction operations (all passes) | **86** (Pass 2: 63 · Pass 3 addendum: 6 · Stage C: 17) |
| Final events (corrected view) | **70** — DRIVER_NO_HELMET 27 · BOTH_NO_HELMET 10 · PILLION_NO_HELMET 3 · NO_VIOLATION 6 · INSUFFICIENT_VISUAL_EVIDENCE 24 |
| License plates | **0 recorded** — none confidently readable; all `NOT_ASSESSED`, nothing invented |

**Audit trail (three passes, all documented):**
- **Pass 2:** 220 programmatic findings triaged; 258 contact-sheet cells visually judged
  (CORRECT 161 / ERROR 9 / UNCERTAIN 88); all 88 uncertain cases re-judged at full resolution —
  73 of 73 apparent helmet-state flips proved to be review noise, 0 were real.
- **Pass 3:** fresh disjoint 168-frame visual sample. Blended frame error 39.3%, dominated by
  rider-subtrack bounding-box placement (46 of 66 errors); detection/track-layer-only error
  11.9%; helmet-state values were correct at every genuinely coupled check. Methodological trap
  recorded for any future reviewer: helmet state may only be judged at frames with a genuinely
  coupled head detection (`time_since_update == 0` + coupled detection); a subtrack's
  largest-box frames can be stale Kalman propagations.
- Every annotation carries provenance (`AI_DIRECT` / `AI_TRACK_PROPAGATED` / `AI_INTERPOLATED` /
  `MANUAL_CORRECTION` / `UNCERTAIN`). Package: `data/dual_camera_test_v1/annotations/`
  (40 files, SHA-256 manifest 39/39 verified).

## 2. Frozen ground-truth release (Stage E)

- Release: **`dual_camera_gt_v1.0_ai_assisted_audited`**, frozen at
  `data/dual_camera_test_v1/releases/dual_camera_gt_v1.0_ai_assisted_audited/`
  (44 files, checksums 43/43 verified repeatedly, including on the Mac).
- Contents include the validity layer (14,249 rows, exactly 6,727 spatially valid) and the
  uncertainty layer (178 rows). One event differs from the working package view:
  `pair_020_moto_01` is FRONT_VIEW_ONLY in the frozen release (the excluded "motorcycle" proved
  to be a man on a bicycle loaded with orange cargo sacks). Event total unchanged at 70.
- The release has **never been modified after freezing** — not during Stages F–L, not during the
  post-completion audit.

## 3. Synchronization engine design (Stage F)

- Design of record: `docs/phase12/INTEGRATED_DUAL_CAMERA_SYNCHRONIZATION_ENGINE_DESIGN.md`.
- Forensic re-examination found **no sign error** in the existing code. Canonical time =
  front_pts_norm + offset_sec, with the back camera as the reference timeline.
- Clock drift (up to ±76 ms; pairs 004/005/008/015) is **reported, never applied**, so
  near-frame residuals are lower bounds. No physical genlock or exposure-level synchronization
  was ever established for this footage — a property of the source data, not of the code.

## 4. Integrated pipeline implementation (Stage G) and tests (Stage H)

- `src/pipeline/dual_camera_pipeline.py` (1,927 lines), implemented to the Stage F design;
  internal smoke 49/49 PASS. Model weights SHA-256-pinned in the tracking configuration; class
  mappings enforced at three layers; each model constructed once and shared across cameras.
- Stage H: **54 new tests, all passing.** Full repository suite: **140 passed / 3 failed**, the
  3 failures being pre-existing environment issues identical before and after (missing
  `soundfile`, missing `easyocr`, an Apple-Silicon MPS test running under Linux). No failure is
  attributable to the delivered pipeline code.
- Model identities (SHA-256, verified on VM and Mac):
  - Phase 9 helmet model `results/phase9_multiseed/exp_combined/seed_42/weights/best.pt` =
    `0929a882667e0f7e2b631071ab218a2415cc1b816cdede7fb6812543d840b0e5`
    (classes 0 = no_helmet, 1 = helmet)
  - `yolov8n.pt` = `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`
    (COCO person = 0, motorcycle = 3)

## 5. Single-pair proof (Stage I) and full 25-pair run (Stage J)

- **Stage I:** the pipeline reproduced Pass 1 **exactly** on pair_001 — delta 0 in every
  category, both cameras (CPU 160.7 s, peak RSS ≈ 981 MB).
- **Stage J (2026-11-09):** full run on Kaggle T4 (account `elafahmed6563`), kernel wall 958 s
  (≈0.27 GPU-hours; quota read directly afterwards: **25.01 GPU-hours remaining**, refresh
  2026-11-14). **25/25 pairs passed all gates.** Detections identical to Pass 1 — delta 0 in all
  200 pair×camera×category cells; 15,241 frames exact; drift values reproduce Phase 11.
- The pipeline derived **85 events** from uncorrected inputs vs **70** in the corrected release;
  that gap is precisely the correction layer, and Stage K decomposed it completely.

## 6. Evaluation (Stage K) — frozen protocol, zero amendments, no tuning

Protocol frozen before any metric was computed (SHA `466f5f31…`); pair-bootstrap B = 2000.

| Metric family | Result |
|---|---|
| Head detection | P 0.689 / R 1.000 / F1 0.816 · AP50 0.731 |
| Motorcycle track-level | P 0.82 / R 1.0 / F1 0.901 (all 18 FP tracks = the correction-excluded set) |
| Rider (spatially valid subset) | F1 / IoU 1.000 |
| Helmet, observation-level | Perfect on 6,753 decided observations |
| Helmet, track-level | **64.1%** headline — see §6.2 for the forensic explanation |
| Identity (IDF1) | 0.9987 · 0 ID switches |
| Cross-camera association | Motorcycle F1 0.909 · abstention 38.8% (orphans 63.3%) |
| Events | P 0.753 / R 0.914 / F1 0.826 · timing errors 0.0 s · all 27 error instances decomposed, residual unexplained = 0 |
| Declared NOT COMPUTABLE (never fabricated) | Person detection metrics · frame-level motorcycle AP · HOTA · item-level rider association |

**Governing caveat (applies to every number above):** the ground truth was built from Pass 1
detections plus the 86 documented corrections, and Stage J bit-reproduces Pass 1 — so coupled
observation-level agreement is perfect **by construction**. The independent correctness evidence
is the visual audit material (Pass 3, Stage B, Stage D). This caveat is disclosed in every
evaluation document.

### 6.1 Frame-count reconciliation (post-completion Task 3)

An earlier Phase 12C report stated 15,637 decoded frames; all processing stages used 15,241.
**Verdict: prose-only error.** Phase 12C's own forensic CSV sums to 15,241; four independent
counting conventions (ffprobe packets, ffprobe `nb_frames`, OpenCV frame-count property, OpenCV
sequential decode) all total exactly 15,241 and agree per-video on 50/50 files; excluded frames
= 0 everywhere. **No decodable source frame went unprocessed by Pass 1 or Stage J; the
unprocessed set is empty.** The two Phase 12C reports were left unedited (frozen records); the
correction of record is `docs/phase12/PHASE12_FRAME_COUNT_RECONCILIATION.md` with
`results/phase12_postcompletion/frame_count_reconciliation.csv` (50 rows).

### 6.2 Helmet track-level failure analysis (post-completion Task 4)

Forensic breakdown of all 209 decided rider tracks (AGREE 134 · PIPELINE_UNCERTAIN 73 ·
FALSE_HELMET 1 · FALSE_NO_HELMET 1 — reproducing Stage K exactly):

- **The 64.1% is an aggregation/export artifact, not a classification failure.** The exported
  track state is read from a 10-frame sliding-window EMA at end-of-run; frames after a
  subtrack's last coupled observation keep appending "undetected", so any subtrack whose last
  coupling ends ≥10 stepped frames before its parent track ends exports zero mass → UNCERTAIN.
  Separation is perfect in the data: all 73 abstentions have tail gap ≥ 10 (median 66, max 285);
  all agreed tracks < 10.
- **Where the pipeline decided, agreement is 134/136 = 98.5%.** In 72/73 abstentions the
  pipeline's own observation majority already matched the ground truth (65 unanimous). Worst
  case: a pair_024 pillion with 84/84 unanimous no-helmet observations discarded over a
  226-frame tail gap.
- The 2 hard disagreements are both pair_004 front orphans: `front_orphan_0003` (a genuine
  mid-track helmet-donning transition; GT not endorsed for the donning window) and
  `front_orphan_0005` (sparse detector error; GT rests on a Stage C visually verified constant).
- Driver/pillion identity confusion: **none** (0/23 multi-rider parents). Track fragmentation
  in the scored data: **none**.
- Full report: `docs/phase12/HELMET_TRACK_LEVEL_FAILURE_ANALYSIS.md`; per-track CSV (209 rows);
  18 evidence PNGs in `docs/phase12/track_failure_examples/`; 8 worked examples.

### 6.3 Human validation protocol (post-completion Task 5) — designed, NOT executed

`docs/phase12/HUMAN_VALIDATION_PROTOCOL.md` (538 lines): census of all 73 pipeline-uncertain
tracks, both hard disagreements, all 46 decided events' roles, all 70 events re-marked for
onset/end, all LINKED + AMBIGUOUS cross-camera links and all 30 low-evidence subtracks, plus a
powered random sample of 60/134 agreeing tracks — ≈399 decision units, ≈55 person-hours, two
blinded independent reviewers, 40-item calibration set with κ ≥ 0.70 entry gate, pre-registered
thresholds, hash-frozen sample manifest held out from tuning. Human labels, if ever collected,
must land as a separate provenance layer and any corrections as a new v1.1 release.
**No execution record exists** (see the owner's statement at the head of this report).

## 7. Forensic audit of the whole mission (Stage L) — CLEAN, 0 discrepancies

- 50/50 videos SHA-256-identical to the manifest; Phase 11 metadata 6/6 byte-identical to the
  pristine source tarball; code vs tarball 74/74 with exactly the 5 expected new files;
  Pass 1 outputs 301/301 untouched; annotation package 39/39; release 43/43.
- Final Stage L deliverables: `docs/phase12/INTEGRATED_SYSTEM_{FINAL,PERFORMANCE,
  SYNCHRONIZATION}_REPORT.md`, `INTEGRATED_SYSTEM_RESEARCH_LIMITATIONS.md`,
  `results/phase12_integrated/` (checksums 8/8).

## 8. Delivery to the Mac repository and GitHub (post-completion Tasks 1, 2, 6, 7)

- Sync package v2: **530 files, 22,521,615 bytes**, `unzip -t` clean, internal SHA-256 manifest
  covering 529 files; all 136 files overlapping the working mirror verified byte-identical
  before packaging. Transferred intact into the Mac repository and unzipped there.
- **Checksum verification executed on the Mac:** release 43/43 OK · annotation package 39/39 OK ·
  integrated results 8/8 OK · both model weight hashes exact · Stage J 25/25 pair directories
  present · integrated pipeline present. Videos and weights deliberately excluded from the
  package (already on the Mac, git-ignored); no caches or credentials transferred.
- **Git:** commit **`1a1b3b26c6c7d09d1fccf5daefaf678a37be813e`** on `main` (parent `6aba044…`),
  530 files changed, 638,965 insertions, purely additive. The first push failed on a missing
  upstream configuration; the corrective `git push --set-upstream origin main` succeeded and was
  **verified independently on GitHub** (HEAD = `1a1b3b2…`) and against the Mac's own remote ref.
  No force-push was ever used; history is strictly extended.
- Full delivery report: `docs/phase12/POST_COMPLETION_REPRODUCIBILITY_AND_VALIDATION_REPORT.md`.

---

# PART B — WHAT STILL NEEDS TO BE DONE

## 9. Immediate housekeeping (owner, minutes)

1. **Commit the two closing reports** (both already sitting in the Mac repo, uncommitted):
   ```bash
   cd /Users/sayemuddin/Desktop/research_helmet
   git add docs/phase12/POST_COMPLETION_REPRODUCIBILITY_AND_VALIDATION_REPORT.md
   git commit -m "Helmet: post-completion reproducibility and validation report"
   git push
   ```
   (This master report itself can be added to the same or a following commit.)
2. **Optional tidy-up:** `helmet_mac_sync_postcompletion_2026-11-12.zip` and the old `_chunks/`
   folder in the repo root were deliberately left uncommitted; delete them whenever convenient.
   The `.venv-mac/` folder was never created (environment install waived) — nothing to clean.

## 10. Scientific items that remain open (owner's decisions, none started)

3. **Human-verification status of the ground truth.** The release label in the repository is
   `AI_ASSISTED_AUDITED`. The owner has stated (2026-11-12) that he has personally verified all
   annotations. For the repository and any publication to carry a human-verified claim, the
   designed protocol (`HUMAN_VALIDATION_PROTOCOL.md`) specifies the required completion record:
   reviewer identities, dates, hash-frozen sample manifest, and measured agreement metrics —
   after which corrections, if any, are issued as a **v1.1 release** (v1.0 is never edited in
   place).
4. **Track-state export fix (identified, deliberately not applied).** Export each track's helmet
   state from its last coupled observation (or report the observation majority alongside the EMA
   state). Expected effect per the Task 4 analysis: track-level coverage rises from 136/209
   decided to ~208/209, with decided agreement at 98.5%. Requires re-running export + Stage K
   track-level scoring under a new frozen protocol amendment.
5. **Phase 12C prose erratum.** Two frozen Phase 12C reports still contain the incorrect 15,637
   figure in prose. The reconciliation report is the correction of record; whether to also
   append an errata note to those files is the owner's editorial decision.
6. **NOT COMPUTABLE metric families.** Person detection metrics, frame-level motorcycle AP,
   HOTA and item-level rider association remain uncomputed; each needs either additional
   ground-truth structure or a different evaluation design. They must not be estimated from the
   existing data.
7. **Mac-side test execution.** The pipeline test suites have never been run on macOS (the Mac's
   Homebrew Python 3.14 lacks pytest and the project environment; the owner waived installation
   on 2026-11-12). If a Mac run is ever wanted: create a Python 3.11 venv (the version
   `requirements.txt` was verified on), `pip install -r requirements.txt`, and run the three
   `tests/test_phase12_pipeline_*.py` files. The VM record stands until then.
8. **Publication and next-phase decisions** (venue, submission, any OCR / training / new-data
   work) were explicitly out of scope for the completed missions and remain entirely the
   owner's. Nothing has been started, submitted or published.

## 11. Standing limitations to carry into any future work or publication

- Ground truth is model-bootstrapped (Pass 1 + 86 documented corrections); coupled metrics are
  self-consistency measures — the Phase 12B protocol tension is disclosed, not resolved.
- Clock drift up to ±76 ms is reported, never compensated; no genlock exists in the source
  footage; near-frame sync residuals are lower bounds.
- Cross-camera association abstains on 38.8% of rider observations (63.3% of orphans) — an
  abstention-honest design, at a real coverage cost.
- License plates are `NOT_ASSESSED` throughout; no plate text exists in the dataset.
- Kaggle quota at last direct read (2026-11-09, post-Stage J): 25.01 GPU-hours remaining,
  refresh 2026-11-14 — re-check before planning any new GPU run.

---

*End of handover. All figures in this report are traceable to the cited project files, checksum
manifests, and the GitHub commit history. Where a claim could not be evidenced, it is stated as
unexecuted, uncomputed, or recorded as the owner's statement — nothing in this report upgrades a
claim beyond its evidence.*
