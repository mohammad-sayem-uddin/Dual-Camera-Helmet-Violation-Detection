# Post-Annotation Stage J — Full 25-Pair End-to-End Inference Run

**Status: COMPLETE — 25/25 pairs passed every gate. Date: 2026-11-09/10 (EST/UTC).**
Label of all outputs: AI-assisted pipeline output — NOT human-verified ground truth.
Evaluation against the frozen release is Stage K's task and is deliberately NOT
performed here; this stage reports structural validation and a Pass 1
reproduction comparison only.

## 1. Venue and execution

- Venue: **Kaggle T4** (Tesla T4, torch 2.11.0+cu130, ultralytics 8.4.208),
  account **elafahmed6563**, kernel `elafahmed6563/helmet-stagej-pipeline-full-run`
  (v1; pushed 2026-11-10 00:03 UTC, RUNNING immediately, COMPLETE 00:20:25 UTC).
- Kernel wall time: **958 s** (~16 min). Sum of per-pair pipeline wall times:
  675.7 s; max per-pair peak RSS 2,164.5 MB (well inside the bounded-memory design).
- The pipeline ran via its own CLI (`python -m src.pipeline.dual_camera_pipeline
  --output-root /kaggle/working/stagej_out --device cuda`), one isolated worker
  subprocess per pair, exactly as designed in Stage F and implemented in Stage G.
- **No pipeline code, weights, configs, approved modules, frozen package/release,
  or tests were changed.** The only code fix in this stage was inside the Stage J
  validation script itself (a module-level `nonlocal` syntax error), which is
  Stage J tooling, not pipeline code.

## 2. Quota disposition (pre-flight gate 2)

The installed Kaggle CLI exposes a `quota` command (Stage I's earlier check
predated it), so the quota was read directly — no caveat-proceed was needed:

- Pre-flight: GPU used 4.72 h / **remaining 25.28 h** of 30.00 h (refresh 2026-11-14).
- Post-run: GPU used 4.99 h / remaining 25.01 h.
- **Stage J consumed ≈ 0.27 GPU-hours**, in line with the ≈0.3–0.5 h estimate.
- The kernel was never queued or rejected on quota grounds.

## 3. Staging route (pre-flight gates 1 and 3)

Lightest reliable route, per the Stage I §5 plan — the 2.1 GB video dataset was
NOT re-uploaded:

- Existing dataset `elafahmed6563/helmet-annotation-inputs` (2,182,327,905 B,
  status ready; auth live-checked pre-flight) already extracts to a `repo_code/`
  tree containing the videos, metadata, weights, configs and all approved `src/`
  modules at HEAD 6aba044. The only missing piece was `src/pipeline/` (Stage G).
- A tiny overlay dataset **`elafahmed6563/helmet-pipeline-overlay`** was built
  carrying only `src/pipeline/__init__.py`, `src/pipeline/dual_camera_pipeline.py`
  and an `overlay_manifest.json` of per-file SHA-256 hashes. (v1 of the upload
  silently skipped the `src/` folder; v2 with `--dir-mode zip` fixed it — the
  extracted tree was verified via `datasets files` before the kernel was pushed.)
- The kernel copies the overlay into the repo tree and asserts a **triple hash
  match** (overlaid file == overlay manifest == the Stage G recorded hashes
  `dual_camera_pipeline.py` e0f2204b…, `__init__.py` 1072c99a…) before running.
- Staged weight hashes were asserted in-kernel before the run and recorded in the
  run manifest: `best.pt` sha256 `0929a882667e0f7e2b631071ab218a2415cc1b816cdede7fb6812543d840b0e5`,
  `yolov8n.pt` sha256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`
  — both equal the recorded Stage I values. Gate G7 re-verified these hashes from
  every pair summary after download.
- Output hygiene (Pass 1 ballooning lesson): the kernel zipped ONLY the clean
  `stagej_out/` tree and deleted the staged repo/input copies before exit.
  Downloaded outputs total 55 MB.

## 4. Per-pair gates — 25/25 PASS

Gates (validator `work/stageJ/validate_stagej.py` → `validation_results.json`):
G1 summary status COMPLETE; G2 pipeline validation all_passed (8/8 incl. the
all-passed flag); G3 summary frames_decoded == frames_expected_container ==
Phase 12C forensic count (independent source: frozen package
`frame_coverage_manifest.csv` row counts); G4 detections JSONL lines == frames
per camera; G5 all 13 per-pair artifacts present; G6 sync diagnostics carry
`drift_ms` AND `q5_overlap_lookup_stats` (GAP-1/GAP-4 fields); G7 staged weight
hashes == recorded values. Run manifest reconciles **25/25 COMPLETE**.

| Pair | Wall (s) | Drift (ms) | ε_sync (ms) | Events | Gates |
|------|---------:|-----------:|------------:|-------:|:-----:|
| pair_001 | 37.7 | 0 | 3 | 2 | PASS |
| pair_002 | 31.4 | +2 | 3 | 2 | PASS |
| pair_003 | 29.6 | 0 | 3 | 4 | PASS |
| pair_004 | 17.1 | +36 | 18 | 3 | PASS |
| pair_005 | 24.9 | +70 | 35 | 2 | PASS |
| pair_006 | 18.1 | −12 | 6 | 4 | PASS |
| pair_007 | 18.0 | −8 | 4 | 6 | PASS |
| pair_008 | 19.3 | −76 | 38 | 1 | PASS |
| pair_009 | 28.6 | 0 | 3 | 11 | PASS |
| pair_010 | 17.6 | +14 | 7 | 5 | PASS |
| pair_011 | 22.5 | 0 | 3 | 5 | PASS |
| pair_012 | 30.2 | 0 | 3 | 10 | PASS |
| pair_013 | 42.9 | 0 | 3 | 3 | PASS |
| pair_014 | 23.9 | 0 | 3 | 4 | PASS |
| pair_015 | 21.9 | +74 | 37 | 3 | PASS |
| pair_016 | 33.1 | −20 | 10 | 2 | PASS |
| pair_017 | 17.7 | 0 | 3 | 2 | PASS |
| pair_018 | 15.8 | 0 | 3 | 1 | PASS |
| pair_019 | 25.2 | −16 | 8 | 4 | PASS |
| pair_020 | 29.6 | 0 | 3 | 3 | PASS |
| pair_021 | 31.5 | 0 | 3 | 1 | PASS |
| pair_022 | 27.9 | 0 | 3 | 2 | PASS |
| pair_023 | 32.4 | 0 | 3 | 2 | PASS |
| pair_024 | 52.7 | 0 | 3 | 1 | PASS |
| pair_025 | 26.1 | 0 | 3 | 2 | PASS |

Corpus coverage: front **6,557** frames, back **8,684** frames — exactly the
Phase 12C forensic totals (15,241). Drift values reproduce the Phase 11
measurements (pair_004 +36, pair_005 +70, pair_008 −76, pair_015 +74 ms) and, per
GAP-1, are **reported, never applied**; near-frame ε_sync residuals are lower
bounds on end-of-pair misalignment, as disclosed since Stage F.

## 5. Pass 1 reproduction comparison (structural, not evaluation)

Per-category detection totals were counted identically from both runs' JSONL
bundles (Stage J `num_motorcycles`/`num_persons`/`num_helmet`/`num_no_helmet`
fields vs Pass 1 flat detection lists counted by class name), for every
pair × camera × category cell (200 cells):

- **Every one of the 200 deltas is exactly 0.** No pair was flagged under the
  material-divergence rule (|delta| > 2% of the Pass 1 count AND ≥ 5 boxes);
  no investigation was required.
- Corpus totals (both runs): front motorcycle 7,558 / person 11,673 /
  helmet 1,307 / no_helmet 5,227; back motorcycle 4,598 / person 10,677 /
  helmet 991 / no_helmet 2,280 — **44,311 detection boxes in total**, matching
  Pass 1 exactly. The T4 run reproduces the Pass 1 (Kaggle T4) and Stage I
  (CPU, pair_001) detection behaviour bit-for-bit at count level.
- Motorcycle cross-camera association outcomes are likewise identical to
  Pass 1: LINKED 12 / UNMATCHED 40 / AMBIGUOUS_ABSTAIN 6 /
  INSUFFICIENT_EVIDENCE 27. Rider-track outcomes: LINKED 4 / UNMATCHED 6 /
  ABSTAIN 1 / INSUFFICIENT 9; orphan-track outcomes: LINKED 1 / UNMATCHED 43 /
  ABSTAIN 20 / INSUFFICIENT 56.

## 6. Derived events (pipeline output — evaluation deferred to Stage K)

The pipeline derived **85 events**: DRIVER_NO_HELMET 27, BOTH_NO_HELMET 9,
PILLION_NO_HELMET 4, NO_VIOLATION 6, INSUFFICIENT_VISUAL_EVIDENCE 39.

This count is the pipeline running on **uncorrected** detections/tracks. The
frozen release carries **70 events**, derived from the corrected annotation view
(Pass 2/3 + Stage C corrections excluded false-positive tracks and re-grounded
links before event derivation). The 85-vs-70 difference is therefore expected
and is precisely what Stage K will quantify as detection/event-level agreement;
no accuracy claim is made here.

## 7. Failures and retries in this stage

- Overlay dataset v1 silently omitted the `src/` folder (CLI default dir mode);
  detected by listing the dataset's files before pushing any kernel; fixed in v2
  with `--dir-mode zip`. No kernel run was wasted.
- Output download attempt 1 aborted mid-transfer (connection reset by peer;
  only `run_info.txt` and a partial `_logs/` landed). Attempt 2 via a fresh
  systemd unit completed with exit 0; `stagej_outputs.zip` verified with
  `unzip -t` (no errors).
- The Stage J validator script itself had a syntax error on first run (see §1);
  fixed in the tooling only.
- The kernel itself ran once, first try, with no errors, no queueing, and no
  retries.

## 8. Artifacts

- Outputs: `work/stageJ/outputs/` — `stagej_out/` (25 pair dirs × 13 artifacts +
  `run_manifest.json`, `run_config.json`, `run_log.txt`, `_logs/`),
  `stagej_outputs.zip`, `run_info.txt`, kernel log
  `helmet-stagej-pipeline-full-run.log`.
- Validation: `work/stageJ/validation_results.json`, `work/stageJ/logs/`.
- Progress record: `work/stageJ/STAGE_J_PROGRESS.md`.

**Stage J verdict: COMPLETE.** All 25 pairs processed end-to-end on the T4 with
the tested Stage G pipeline and the Stage I-verified models; every structural
gate passed; the run reproduces Pass 1 detection and association behaviour
exactly. Next: Stage K (evaluation vs the frozen Stage E release).
