# Post-Annotation Stage H — Test Report

**Stage:** H (unit + regression + synthetic sync battery + real-video tests) of the
post-annotation mission, testing the Stage G pipeline
`src/pipeline/dual_camera_pipeline.py` against the Stage F design
(`INTEGRATED_DUAL_CAMERA_SYNCHRONIZATION_ENGINE_DESIGN.md`, §7 + §7.13 test matrix).

**Status: COMPLETE.** All Stage H obligations are met. The full suite stands at
**140 passed / 3 failed**, where the 3 failures are exactly the pre-existing
environment failures of the baseline (same node IDs, proven below). **No defect
was found in the Stage G implementation**; the only fix made during Stage H was
to an arithmetic error in one of Stage H's own new tests (S-DROP, §5).

**New test files (the only repo test changes; no existing test was edited):**

| File | Tests | Runtime |
|---|---|---|
| `tests/test_phase12_pipeline_synthetic.py` | 16 | 23.70 s |
| `tests/test_phase12_pipeline_semantics.py` | 14 | 7.54 s |
| `tests/test_phase12_pipeline_real_video.py` | 24 | 111.69 s |
| **Total new** | **54, all passed** | **142.93 s** |

Execution note: a first Stage H attempt died ~3 minutes in with no deliverables.
This run was checkpointed section-by-section in `work/stageH/STAGE_H_PROGRESS.md`,
with all pytest output redirected to files under `work/stageH/`
(`baseline_pytest.txt`, `run_synthetic2.txt`, `run_semantics.txt`,
`run_real_video.txt`, `run_full_regression.txt`).

---

## 1. Baseline and full regression

| Run | Command scope | Result | Runtime |
|---|---|---|---|
| Baseline (before Stage H files) | `tests/` (89 pre-existing tests) | **86 passed / 3 failed** | 154.41 s |
| Full regression (after Stage H) | `tests/` (143 tests) | **140 passed / 3 failed** | 243.47 s |

The failure set is identical before and after — the same three pre-existing
environment failures, by node ID:

1. `tests/test_environment_readiness.py::test_core_dependencies_import` — missing `soundfile` in the VM venv.
2. `tests/test_environment_readiness.py::test_apple_silicon_mps_support` — Apple-Silicon MPS test on a Linux VM.
3. `tests/test_model_adapters.py::test_easyocr_adapter_text_cleaning_and_read` — missing `easyocr` (OCR is out of scope by standing restriction).

140 = 86 baseline passes + 54 new passes. No existing test was weakened,
skipped, or edited.

---

## 2. Test inventory mapped to the design matrix (§7.13)

### 2.1 Synthetic battery — `test_phase12_pipeline_synthetic.py`

| Design ID | Test node | What is asserted |
|---|---|---|
| S-6060 | `test_s6060_aligned_pair_primary_and_tick` | 60/60 pair: primary loop COMPLETE, detector exactly-once per (camera, frame) 0..59 both cameras; tick mode skips nothing (unique updates 60/60, skipped duplicates 0/0, tracker `frames_updated` 60/60). |
| S-6030 | `test_s6030_primary_loop_physical_frame_coverage` | 60/30 pair, offset +0.5: back contributes its 45 **physical** frames (not ~90 tick presentations); front 90; coverage complete both cameras. |
| S-NEW-1 | `test_s_new_1_tick_mode_60_30_unique_updates` | Tick mode: back unique updates == 30 (physical back frames in overlap, recomputed independently from the PTS tables), front unique == ticks (59); skipped back duplicates == ticks seen − unique; tracker advanced exactly once per unique frame; detector call set == recorded detection set. |
| S-NEW-2 | `test_s_new_2_frozen_frame_injection_tracker_guard` | Same `frame_index` re-presented 5× at the real tracker: `did_advance_state` true once, `duplicate_ticks_skipped` == 4, `frames_updated` frozen, active-track state hash unchanged. |
| S-NEW-3 | `test_s_new_3_vfr_burst_gate_updates_equal_distinct_indices` | Burst presentation with interleaved repeats through `DeduplicatedTemporalGate`: update executions == distinct indices (10 of 14 presentations), repeats return the cached record, counters reconcile. |
| S-NEW-4 | `test_s_new_4_tick_primary_detection_equivalence` | Tick-mode vs primary-loop detection records equal by (camera, frame_index) over the overlap (content-scoped comparison, §4), tick set ⊆ primary set. |
| S-VFR | `test_s_vfr_mixed_intervals_index` | Injected PTS table with 60 fps / 30 fps / 120 fps burst stretches: `is_vfr`, interval-class counts, monotone `nearest_frame`, non-trivial nominal-grid deviation. |
| S-DROP | `test_s_drop_pts_gap_index` | Two dropped frames merge into one 50 ms gap: packet count drops by exactly 2, a target at a dropped frame's time resolves to a neighbour at half-gap delta (16.67 ms). |
| S-DUP-PTS | `test_s_dup_pts_duplicate_ticks_not_deduplicated` | GAP-5 fact pinned: duplicate ticks are retained and counted as packets (5 packets, 4 distinct timestamps); lookup at the duplicated time is exact. |
| S-OFFSET-013 | `test_s_offset_013_worked_example` | Synthetic pair with offset **+6.695** (initial +8.357 / final +4.595 in metadata): operative offset is `offset_sec`; canonical shift front +6.695 / back 0; lookup at t_ref 7.195 targets front-stream 0.500 and returns frame 5; canonical = actual + offset. |
| S-OFFSET-POS | `test_s_offset_pos_small_positive` | Offset +0.25: overlap starts at 0.25 on the back timeline; front canonical shift +0.25. |
| S-OFFSET-NEG | `test_s_offset_neg_front_leads` | Offset −2.380: overlap starts at back time 0, ends at −2.380 + 2.9 = 0.52; at t_ref 0 the front target is +2.380 s (nearest frame 24 at 2.4 s). |
| S-OVERLAP-EDGE | `test_s_overlap_edge_boundaries` | `get_synchronized_pair` at overlap start/end: `is_in_overlap` true (and `valid_pair` at start); one frame interval outside either edge: false. |
| S-MISSING-CAM | `test_s_missing_cam_failed_precondition_no_artifacts` | Missing back camera: `run_pair` → `FAILED_PRECONDITION` / `ERR-PRECONDITION`, and the pair output directory is never created (design §7.11 "no artifacts"). |
| S-MISSING-CAM (batch) | `test_s_missing_cam_batch_isolation_run_all` | `run_all` with **real detectors** in subprocesses: the broken pair fails `FAILED_PRECONDITION` first (sorted order), the intact pair still reaches COMPLETE, `run_manifest.json` pair table records both correctly. |
| S-DET-1 | `test_s_det_1_determinism_content_scoped` | Same pair run twice: coverage CSVs **byte-identical**; detections / track-frames / association / events / tracks-final / sync-diagnostics / summary **content-identical** after the §4 normalization (summary compared outside its design-exempt runtime fields `wall_time_sec`, `peak_rss_mb`). |

### 2.2 Pipeline unit semantics — `test_phase12_pipeline_semantics.py`

| Obligation | Test node | What is asserted |
|---|---|---|
| D1 canonical mapping | `test_canonical_mapping_and_drift_never_applied` | On the coverage artifacts of a real run: every front row's `canonical − pts_norm` == +0.5 exactly (first and last frame alike); every back row's == 0. |
| GAP-1 | (same test) + `test_diagnostics_field_separation_gap4_gap6` | Operative offset is `offset_sec` (0.5), not initial (0.65) / final (0.42); `drift_ms` == 74.0 appears in summary **and** diagnostics, while the canonical shift never deviates from the constant offset — drift is reported, never applied. |
| GAP-4 | `test_q5_true_lookup_vs_nominal_grid_gap4` | Two-rate stream (30 fps → 60 fps): nominal-grid deviation max **> 500 ms**, true Q5 lookup error max **< 20 ms** over the same stream — the two quantities are demonstrably different and separately named. |
| GAP-4 (lookup direction) | `test_q5_lookup_target_uses_offset` | With offset = half a frame, Q5 front mean ≈ 0 and back mean ≈ 8.33 ms — proving front targets are computed as `t_ref − offset_sec` (D1 lookup direction). |
| GAP-4 / GAP-6 (artifacts) | `test_diagnostics_field_separation_gap4_gap6` | Diagnostics carry `offset_sec`, `epsilon_sync_ms`, `drift_ms`, `sync_quality` as separate fields; no `accuracy_classification` key; grid quantity present only as `nominal_grid_deviation_ms` inside the per-stream blocks, with a field definition stating it is NOT Q5; Q7 disclaimer verbatim. |
| §7.7 resume lifecycle | `test_checkpoint_resume_lifecycle` | Six phases: complete → skip with zero detector calls → tampered recorded hash re-runs → summary not COMPLETE re-runs → `resume=False` re-runs → unparseable summary re-runs cleanly. |
| §7.11 taxonomy surface | `test_error_taxonomy_surface` | Exactly the 8 design error codes exist; `PipelineError` rejects unknown codes; `PairStatus` has exactly the 6 design states. |
| ERR-PRECONDITION | `test_err_precondition_variants` | Five variants — missing pinned metadata, wrong metadata filename (GAP-7 pinning), missing tracking config, missing association config, unknown pair id — each `FAILED_PRECONDITION` with no pair directory created. |
| ERR-DETECTION | `test_err_detection_on_detector_exception` | Detector raising in `detect_frame` → `FAILED_RUNTIME` / `ERR-DETECTION`, recorded in `pair_status.json`. |
| ERR-COVERAGE | `test_err_coverage_on_bundle_pts_mismatch` | Detector bundle whose PTS disagrees with the PTS table → `FAILED_RUNTIME` / `ERR-COVERAGE` (invariant I2 gate). |
| ERR-TRACKING | `test_err_tracking_on_tracker_init_failure` | Tracker factory raising → `FAILED_RUNTIME` / `ERR-TRACKING`. |
| ERR-ASSOCIATION | `test_err_association_on_engine_failure` | Association engine raising inside `associate_pipelines` → `FAILED_RUNTIME` / `ERR-ASSOCIATION`. |
| ERR-RESOURCE | `test_err_resource_ceiling` | With the ceiling seam (`DEFAULT_RSS_CEILING_MB`) set to 0, a fully processed pair fails post-validation with `ERR-RESOURCE` instead of being marked COMPLETE. |
| ERR-INTERNAL | `test_err_internal_on_unexpected_exception` | An unexpected `RuntimeError` in diagnostics → `FAILED_RUNTIME` / `ERR-INTERNAL` with traceback preserved. |
| ERR-DECODE | `test_err_decode_on_truncated_stream` | Back video truncated to 55% (moov atom removed; behaviour probed on this stack before the assertion was written) → `FAILED_RUNTIME` / `ERR-DECODE`. |

### 2.3 Real-video tests — `test_phase12_pipeline_real_video.py` (24 cases)

All five Stage H pairs were exercised for real — no skips. Construction is
index/PTS-level through the real reader and the pipeline's `SyncContext`, with
bounded decoding only (midpoint spot checks; one 24-tick scan on pair_015).
No model inference (Stage J scope).

| Test (parametrization) | Pairs | What is asserted |
|---|---|---|
| `test_real_pts_table_integrity` ×5 | 001, 013, 014, 015, 019 | `FramePTSIndex.frame_count` == Phase 12C forensic count from `frame_coverage_manifest.csv` for all 10 streams (206/397, 404/608, 266/276, 339/190, 372/231); `pts_norm[0]` == 0; PTS strictly increasing. |
| `test_real_offset_three_way_agreement` ×5 | all | Pipeline `offset_sec` == metadata `offset_sec` == manifest `temporal_offset_sec` (exact); same for residual (Q4) and drift; quality agrees; canonical shifts per D1. |
| `test_real_overlap_windows` ×5 | all | Overlap window equals the sanctioned formulas **including the reader's tightening step** (see §5.2); duration internally exact and within 0.1 s of the manifest. |
| `test_real_canonical_spot_checks` ×5 | all | At the overlap midpoint: valid pair in overlap; chosen front/back indices == `nearest_frame` at the sanctioned targets (`t_mid − offset` / `t_mid`); reported actual PTS == table values; inter-camera skew < 50 ms. |
| `test_real_pair_013_offset_worked_example` | 013 | Offset +6.695; the front frame nearest 0.500 s has canonical time pts + 6.695 and the reader lookup at that canonical time returns exactly that frame (design §2.2 relations). |
| `test_real_vfr_classification` | 001, 013, 014, 015 | Forensic VFR flags: 001 front, 013 back, 014 back are VFR; pair_015 back is constant-rate at 29.97 fps effective, 190 frames. |
| `test_real_pair_015_unique_frame_counting` | 015 | 24 ticks at 60 Hz from overlap start (fresh reader): `is_new` counts == distinct physical frames presented for **both** cameras; the 29.97 fps back camera genuinely repeats (distinct < 24, ≥ 10). |
| `test_real_q5_and_grid_quantities` | 013, 015 | pair_013 back nominal-grid deviation max > 500 ms (measured 2,531 ms); pair_015 Q5 over the overlap: > 100 samples, both cameras' max < 20 ms, carried separately from Q4 = 37 ms. |

---

## 3. Counts summary

| Battery | Written | Passed | Failed |
|---|---|---|---|
| Synthetic (S-*) | 16 | 16 | 0 |
| Unit semantics | 14 | 14 | 0 |
| Real-video | 24 | 24 | 0 |
| **Stage H new tests** | **54** | **54** | **0** |
| Pre-existing suite (unweakened) | 89 | 86 | 3 (environment, unchanged) |
| **Full suite** | **143** | **140** | **3** |

Total Stage H test runtime (the four measured runs): ≈ 541 s on this VM
(~1.5-core quota), of which the pre-existing suite accounts for ~398 s across
its two runs.

---

## 4. Determinism scoping (S-DET-1) — what "content" means and why

Design §7.8 requires byte-identical per-pair artifacts on re-run, exempting
only `pair_summary.json`'s runtime fields. Two **approved-component**
nondeterminism sources make raw byte equality impossible without falsifying
approved payloads (Stage G interpretation #9 reached the same conclusion for
the first):

1. **Wall-clock latency fields** — `FrameDetectionBundle.to_dict()` and the
   association batch `to_dict()` embed `*_latency_ms` measurements.
2. **Phase 10 alert IDs** — `src/temporal/violation_decision.py` mints
   `VIOLATION-{track}-{frame}-{uuid4 hex[:6]}`, so `*_events.json` and
   `*_track_frames.jsonl` embed a fresh random suffix per run.

S-DET-1 therefore compares: coverage CSVs **byte-for-byte** (they contain
neither source — and they matched), and all JSON/JSONL artifacts after a
normalizer that drops `*_latency_ms` / `runtime_ms` keys and masks the alert-ID
uuid suffix. With exactly those two documented exemptions, **every content
field of every artifact matched across runs**, including the summary outside
its exempt runtime fields. This is a property of the approved payloads, not a
pipeline defect: the pipeline writes approved `to_dict()` output verbatim, as
§7.9 mandates.

---

## 5. Defects found and fixed

### 5.1 Stage G pipeline defects: **none**

No test exposed a defect in `src/pipeline/dual_camera_pipeline.py`. The module
was not modified during Stage H. Integrity was re-verified at the end of the
stage: all 63 `src/` + `configs/` files and all 84 annotation-package +
frozen-release files are byte-identical to the Stage G baseline checksums
(`work/stageG/baseline_*.sha256`, re-run under Stage H: 0 mismatches).

### 5.2 Design-vs-code nuance recorded (not a pipeline defect)

Design §2.2 quotes pair_013's overlap as [6.695, **13.479**] — the *metadata*
window (offset + container front duration 6.784 s). The reader then tightens
the end against the PTS-index durations
(`VideoReader.duration_sec == FramePTSIndex.duration_norm_sec`), giving the
operative window [6.695, **13.410**] (6.695 + 6.715). The pipeline consumes the
tightened window, as it must (design D1: the code is authoritative; the
canonical-time relations of §2.2 are unaffected). Stage H tests assert the
tightened formula explicitly, reproducing both computation steps, with the
manifest duration as a ≤ 0.1 s cross-check. The Stage F design document's
worked example should be read as the pre-tightening window.

### 5.3 Stage H test-side fix (own test, not the pipeline)

`test_s_drop_pts_gap_index` initially built the "dropped" stream with the
wrong interval count (61 intervals instead of 59), so the frame-count
assertion compared 62 == 60. The test's stream construction was corrected
(19 + 1 merged 50 ms gap + 39 intervals = 60 frames); the pipeline/index code
was never at fault. Before/after: synthetic battery went from 15 passed /
1 failed to 16/16.

---

## 6. Method notes and limitations

- **Injected index objects** were used for S-VFR, S-DROP and S-DUP-PTS, as
  design §7.13 expressly allows: OpenCV's mp4v writer cannot produce VFR,
  dropped-packet, or duplicate-PTS containers, so those stream shapes are
  injected at the `FramePTSIndex` level, while every behaviour that needs a
  real container (decode, tick iteration, reader dedup state) is tested on
  generated files or the real corpus.
- **S-DUP-PTS scope:** the test pins the GAP-5 fact (duplicates retained and
  packet-counted). End-to-end divergence detection is the pipeline's coverage
  gate comparing decoded vs packet counts; fabricating a container whose
  decoded count provably diverges from its packet count was not achievable
  with the available encoder, so that path is covered at the gate's other
  failure points (bundle PTS mismatch → ERR-COVERAGE; truncated stream →
  ERR-DECODE).
- **ERR-RESOURCE** is triggered through the module's documented ceiling
  constant (the design names a configured ceiling without a config field —
  Stage G interpretation #5); the check itself (post-validation, pre-COMPLETE)
  is the real code path.
- **Real-video bounds:** Stage H deliberately runs no model inference on real
  videos (Stage J). Real-pair pipeline behaviour is verified at the
  reader/PTS/SyncContext level plus the reader's own synchronized-pair path —
  the same surfaces the primary loop consumes.
- Tests assume the suite is run from the repository root with the project
  venv (`python -m pytest tests/`), matching every pre-existing test file's
  convention.
