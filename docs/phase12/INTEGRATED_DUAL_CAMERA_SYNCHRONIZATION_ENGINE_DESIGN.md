# Integrated Dual-Camera Synchronization Engine — Architecture Audit & Design (Stage F)

**Stage:** F of the post-annotation mission (audit + design only; implementation is Stage G).
**Deliverable designed here:** `src/pipeline/dual_camera_pipeline.py` — an **orchestration layer
around the approved Phase 12C engine**, not a competing synchronization engine.
**Status of this document:** complete audit + complete design. No approved module was modified
and no pipeline code was written in this stage.
**Ground truth for Stage K:** frozen release `dual_camera_gt_v1.0_ai_assisted_audited`
(`data/dual_camera_test_v1/releases/`), labelled `AI_ASSISTED_AUDITED`, never human-verified.

Authoritative sources audited at source level (repo at Phase 12E HEAD):

| Component | File |
|---|---|
| Phase 12C synchronized reader | `src/synchronization/synchronized_reader.py` |
| Phase 12C PTS index | `src/synchronization/pts_index.py` |
| PTS-indexed video reader | `src/video/video_reader.py` |
| Phase 11 sync metadata | `data/dual_camera_test_v1/metadata/synchronization_metadata.json`, `pair_manifest.csv` |
| Phase 12D detector | `src/detection/camera_detector.py` |
| Phase 12D/Phase-10 tracker | `src/tracking/camera_tracker.py` |
| Phase 10 temporal logic | `src/temporal/{temporal_engine,rider_tracker,state_machine,confidence_aggregator,violation_decision}.py` |
| Phase 12E association | `src/association/cross_camera_association.py`, `tracklet_models.py`, `transit_time_model.py` |
| Configs | `configs/tracking_config.yaml`, `configs/cross_camera_association_config.yaml`, `configs/temporal_config.yaml` |
| De-facto orchestration evidence | `work/annotate_pass1.py`, `work/run_pass1_all.py` (Pass 1, Mission 1) |

---

## 1. Executive summary and design decisions

The Phase 12C engine is metrologically careful: it separates the inter-camera sync residual
(`epsilon_sync_ms`) from per-camera PTS discretization (`delta_pts_*_ms`), anchors the reference
timeline to the back camera, and ships a deduplication gate so a repeated frame never advances
temporal state twice. The audit confirms those properties from source (§2–§4) and finds **no
sign error in the code** — but it does find that **the mission brief's formula
`t_front = t_back − Δt_offset` contradicts the code under the natural reading of `Δt_offset`**
(§2.3), that **the reader applies no drift compensation although Phase 11 measured drift up to
±76 ms** (§3.8), and that **Pass 1 — the only full-corpus orchestration ever run — did not use
the synchronized reader at all** (§6). Stage G must be designed against the code as it is, not
against the brief's shorthand.

Binding design decisions (rationale in §7):

- **D1 — Code is authoritative for all time conventions.** `offset_sec` from Phase 11 metadata
  is the single operative offset; the pipeline never recomputes, re-estimates, or re-signs it.
- **D2 — Primary processing loop is per-camera sequential decode** (the Pass 1 pattern, the only
  pattern validated on all 50 videos), with synchronization applied through canonical PTS maps
  and Phase 12C diagnostics. The tick-driven `SynchronizedVideoReader` is retained for
  overlap-window determination, sync diagnostics, previews, and Stage H synthetic tests — it is
  not replaced and not bypassed silently; the design states exactly which of its services the
  pipeline consumes (§7.2).
- **D3 — New-frame-only processing is enforced at three independent layers** (decode cursor,
  reader `is_new_*` flags / `DeduplicatedTemporalGate`, tracker `step()` guard), specified
  purely in terms of PTS identity (§7.5).
- **D4 — One pair per process, summary-file checkpoint, failure isolation** — the Pass 1 driver
  pattern, hardened (§7.7).
- **D5 — Every emitted time quantity carries its name from the §3 taxonomy.** The Pass 1
  nominal-grid deviation must never again be stored in a field named `delta_pts` (§3.9, GAP-4).
- **D6 — No drift compensation in Stage G.** Drift is measured, reported per pair, and carried
  into Stage K's synchronization-residual reporting; compensating it would change Phase 11/12C
  semantics and is out of scope (§3.8, OQ-2).

---

## 2. Audit A — Sign conventions as implemented

### 2.1 The conventions (verified in source)

`SynchronizedVideoReader` (`src/synchronization/synchronized_reader.py`):

- **Reference timeline = back camera.** Class docstring, L133: *"Front camera target time:
  t_front_target = t_ref − global_offset_sec"*; `PairSynchronizationInfo` and the status dict
  set `t_ref_sec == t_back_target_sec` (L241 ff., `get_synchronized_pair(t_back_sec)`).
- **Operative offset = metadata `offset_sec`.** `_load_pair_metadata`, L198:
  `glob_off = float(match["offset_sec"])`, stored as `global_offset_sec`. Verified equal to
  `pair_manifest.csv:temporal_offset_sec` for all 25 pairs (e.g. pair_013: `+6.695` in both).
  The metadata's `initial_offset_sec` / `final_offset_sec` are loaded (L199–200 region) but
  **never used in any time computation** — see §3.8.
- **Front target time:** L246 (in `get_synchronized_pair`): 
  `t_front_target_sec = t_ref_sec − self.sync_info.global_offset_sec`.
- **Canonical (reference-clock) time of a front frame** is therefore the inverse map, and this is
  exactly what Pass 1 used (`work/annotate_pass1.py`, canonical-map block):
  `t_ref = front_pts_norm + offset_sec` (shift applied for `cam == "front"` only; back shift 0).
- **Overlap window on the reference (back) timeline**, L208–209:
  `overlap_start_back_sec = max(0.0, offset_sec)`;
  `overlap_end_back_sec = min(back_duration, offset_sec + front_duration)`;
  then tightened in `__init__` (L167–176) against the PTS-index durations
  (`reader_back.duration_sec`, `offset + reader_front.duration_sec`).
- **Inter-camera PTS skew diagnostic**, L302:
  `inter_camera_pts_skew_ms = |(t_back_actual − t_front_actual) − offset_sec| × 1000`, where both
  actuals are *stream-normalized* PTS — i.e. it measures how far the chosen frame pair sits from
  the exact offset relation. Correct as written.

### 2.2 Worked example — pair_013, `offset_sec = +6.695`

Metadata: front duration 6.784 s, back duration 15.145 s, `offset_sec = +6.695`,
`initial_offset_sec = +8.357`, `final_offset_sec = +4.595` (the operative value is neither —
it is the Phase 11 point estimate `offset_sec`).

- Overlap on back timeline: `[max(0, 6.695), min(15.145, 6.695 + 6.784)] = [6.695, 13.479]`,
  duration 6.784 s — matches `pair_manifest.csv` (overlap 6.784 s). ✔
- A front frame at stream-normalized PTS `0.500 s` has canonical reference time
  `0.500 + 6.695 = 7.195 s` (back timeline). To fetch the front frame for reference time
  `t_ref = 7.195`, the reader targets `7.195 − 6.695 = 0.500` in front-stream time. ✔
- Positive `offset_sec` therefore means: **the front camera's stream clock runs behind the
  back reference clock** — front content appears `offset_sec` later on the back timeline;
  equivalently the front recording started `offset_sec` after the back recording's zero point
  in content terms. Negative offsets (13 of 25 pairs, e.g. pair_019 `−2.380`) mean the front
  content leads: overlap starts at back time 0 and the front target at `t_ref = 0` is
  `+2.380 s` into the front stream.

### 2.3 Reconciliation with the mission brief's formula — CONTRADICTION FOUND AND RESOLVED

The brief (and the original prompt) states: `t_front = t_back − Δt_offset`.

The code computes, for the front *target lookup*, `t_front_target = t_back − offset_sec`.
The two agree **only if `Δt_offset` is defined as `+offset_sec` and `t_front` is read as the
front-stream target time** — *not* as the canonical time of a front frame. Under the other
natural reading (canonical time of a front observation, the quantity association actually
consumes), the correct relation is `t_ref = t_front + offset_sec`, i.e. the brief's formula
with `Δt_offset := −offset_sec`.

**Resolution (binding for Stage G):**

- `Δt_offset` is **defined** as `offset_sec` exactly as stored in Phase 11 metadata
  (`synchronization_metadata.json:offset_sec` ≡ `pair_manifest.csv:temporal_offset_sec`).
- The two sanctioned equations, to be used verbatim in code comments and docs, are:
  - Lookup direction: `t_front_target = t_ref − Δt_offset` (what the reader does, L246).
  - Canonicalization direction: `t_canonical(front frame) = pts_norm_front + Δt_offset`
    (what Pass 1 did and what Phase 12E tracklets consume).
- `t_back` in the brief's formula is the reference time `t_ref`; the back camera needs no
  transform (`t_canonical(back frame) = pts_norm_back`).
- **The code is authoritative.** No sign flip is required anywhere; the ambiguity was in the
  prose, not the implementation. Stage G must include a regression test that pins pair_013's
  worked example above (Stage H, test S-OFFSET-013, §7.13).

### 2.4 Secondary convention findings

- `accuracy_classification` (L211) is derived **only** from `epsilon_sync_ms` with hard-coded
  bands: `<= 10.0 ms` → "Sub-Frame (<= 10.0 ms)", else "Near-Frame (18.0-38.0 ms)". The band
  label is wrong at the margins (an 11 ms or 50 ms residual would still be labelled
  "18.0-38.0 ms"), and it is a *different axis* from the manifest `quality` field: pair_022 is
  MEDIUM quality with a 3 ms residual, pair_019 MEDIUM with 8 ms, pair_004 HIGH with 18 ms.
  Stage G must report residual and quality as separate fields and must not reuse this label
  as a quality claim (GAP-6).
- `residual_uncertainty_ms` on `PairSynchronizationInfo` is a deliberate backward-compatible
  alias of `epsilon_sync_ms` (same value, L202 + dataclass comment). Harmless but a
  double-name for one quantity; the pipeline emits only `epsilon_sync_ms` (§3.3).
- Metadata fallback path: if `data_root/metadata/synchronization_metadata.json` is absent the
  reader silently falls back to `results/phase11_dual_camera/synchronization_metadata.json`
  (L156–160). Both copies currently agree, but Stage G must pin the metadata path explicitly
  in config so provenance is deterministic (GAP-7).

---

## 3. Audit B — The seven time quantities (kept distinct)

The design recognizes the following quantities. Stage G code, Stage J outputs, and Stage K
reports must use exactly these names; no field may silently carry a different quantity
(the §3.9 collision is the cautionary example).

| # | Quantity | Symbol / field | Unit | Source of truth | What it is NOT |
|---|---|---|---|---|---|
| Q1 | Stream (raw) PTS | `pts_raw_sec`, `pts_ticks` | s (ticks: rational `time_base`) | Container packet PTS via ffprobe (`pts_index.py` L104 ff.) | Not comparable across cameras (each stream has its own zero / audio lead) |
| Q2 | Normalized PTS | `pts_norm_sec` | s | `pts_norm = (ticks − ticks[0]) × time_base` (`pts_index.py` L72); frame 0 ≡ 0 | Not a wall clock; not yet cross-camera comparable for the front camera |
| Q3 | Clock offset | `offset_sec` = `Δt_offset` | s | Phase 11 metadata `offset_sec` ≡ manifest `temporal_offset_sec` (reader L198) | Not a per-frame error; a per-pair constant estimate |
| Q4 | Sync residual uncertainty | `epsilon_sync_ms` | ms | Phase 11 `residual_error_ms` (reader L202); per-pair, Appendix A | Not the PTS lookup error of any individual frame (that is Q5) |
| Q5 | Nearest-frame (presentation) error | `delta_pts_front_ms`, `delta_pts_back_ms` (+ signed variants) | ms | Per lookup: `|actual_pts_norm − target|` (`pts_index.py` L168 `nearest_frame`; reader status dict) | Not sync uncertainty; bounded by half the local frame interval of that stream |
| Q6 | Cross-camera transit delay | `delta_transit_sec` (12E) | s | Physical FOV-transit phenomenon modelled by `TransitTimeModel` (`transit_time_model.py`; config: expected 0.75 s, window [−3.50, +6.00] s, σ 1.50 s) | Not a clock error; must never be "corrected" by shifting timestamps |
| Q7 | Exposure synchronization | `exposure_sync_model` | — (qualitative) | **Not established by anything in the stack.** Reader status emits `"container_pts_alignment_not_physical_exposure_genlock"` on every tick | Container PTS alignment does not imply sensor exposure simultaneity or rolling-shutter alignment |

Per-quantity notes:

**Q1/Q2 (raw vs normalized).** `FramePTSIndex.__init__` sorts packet PTS ticks
(`pts_index.py` L58–59) — frame index throughout the stack is **presentation-order index**,
and `VideoReader`'s sequential decoder emits presentation order, so the two agree (validated
by `validate_pts_against_opencv_sequential`, same file). Normalization subtracts the first
packet's PTS, absorbing container start offsets (e.g. audio lead). Stage G must always pass
normalized PTS to detectors/trackers (as `StructuredDetection.pts_timestamp_sec`) and retain
raw PTS in coverage records for forensic reproducibility (Pass 1 did both).

**Q3 (offset).** One constant per pair, range in this corpus: `−2.380 s` (pair_019) to
`+6.695 s` (pair_013). It is an *estimate* produced by Phase 11 audio/correlation alignment;
its uncertainty is Q4, not zero.

**Q4 (residual).** Per-pair values in Appendix A: 21 pairs ≤ 10 ms (sub-frame at 60 fps),
pair_016 exactly 10 ms, and four near-frame pairs — 004 (18 ms), 005 (35 ms), 008 (38 ms),
015 (37 ms). Manifest `quality` is a separate Phase 11 judgement (HIGH ×22; MEDIUM: 008, 019,
022) and does not track Q4 monotonically (§2.4).

**Q5 (nearest-frame error).** A per-tick, per-camera quantity produced by the lookup itself.
On a 60 Hz tick grid, a 29.97 fps stream (pairs 015–018 back) yields `delta_pts_back_ms` up
to ~16.7 ms on ticks between its frames; this is expected discretization, fully distinct from
Q4. Phase 12E consumes *means/maxes* of Q5 via `TimestampQualityRecord`
(`tracklet_models.py` L43–56: `mean_delta_pts_*_ms`, `max_delta_pts_*_ms`, alongside
`epsilon_sync_ms` and `global_offset_sec` — the record's field separation is the model to
follow).

**Q6 (transit delay).** A property of the *scene* (a motorcycle traverses the two cameras'
fields of view at different times), consumed only inside Phase 12E candidate generation and
scoring. `TransitTimeModel` widens its acceptance bounds by a clock-uncertainty term
(`transit_time_model.py` L165 ff.) — the one sanctioned place where Q4-style uncertainty
enters association; the pipeline must pass `epsilon_sync_ms` through `TimestampQualityRecord`
and otherwise keep Q6 strictly downstream of synchronization.

**Q7 (exposure).** No stage of the pipeline may claim genlock. Stage J sync diagnostics must
carry the reader's exposure disclaimer string verbatim, and Stage K's synchronization report
inherits this limitation.

### 3.8 The eighth quantity the reader ignores: clock drift

Phase 11 metadata carries `initial_offset_sec`, `final_offset_sec`, `drift_ms`, and
`drift_rate_a` per pair. Measured `drift_ms` reaches **+36 (pair_004), +70 (pair_005),
−76 (pair_008), +74 (pair_015), −20 (pair_016), −16 (pair_019)** — the same order as, or
larger than, Q4 for the near-frame pairs. The reader loads initial/final offsets but **applies
only the constant `offset_sec`**; no code path applies `drift_rate_a` or a time-varying
offset. Consequences: (a) within-pair sync error can grow toward the pair's ends by up to
~±drift/2 around the point estimate; (b) Q4 as reported understates end-of-pair misalignment
for pairs 004/005/008/015/016/019. This is **GAP-1 (HIGH for Stage K reporting, MEDIUM for
processing)**: Stage G must not silently "fix" it (D6), but Stage J diagnostics must report
`drift_ms` alongside `epsilon_sync_ms` per pair, and Stage K must treat the pair_005/008/015
residuals as lower bounds.

### 3.9 Terminology collision in Pass 1 (must not propagate)

`work/annotate_pass1.py` computes `delta_ms = |pts_norm − idx/effective_fps|` — the deviation
of true PTS from a *uniform nominal grid* — and stores it in `TimestampQualityRecord` fields
named `mean_delta_pts_front_ms` / `max_delta_pts_front_ms`. That quantity is **not** Q5 (no
lookup target exists in the per-camera loop). For VFR streams it conflates VFR jitter with
lookup error. Phase 12E's orphan engine consumes those fields as if they were Q5. Stage G
must (a) compute true Q5 statistics when a lookup exists, (b) name the nominal-grid quantity
`nominal_grid_deviation_ms` if it is reported at all, and (c) document in Stage J outputs
which definition each `delta_pts` field carries (GAP-4).

---

## 4. Audit C — Reader behaviour on the hard cases

Evidence from `synchronized_reader.py`, `video_reader.py`, `pts_index.py`. Verdicts:
**HANDLED** (correct today), **HANDLED-WITH-CAVEATS**, **GAP** (Stage G must design around it;
approved code is not modified).

| # | Case | Behaviour as implemented | Verdict / severity |
|---|---|---|---|
| C1 | VFR streams (pair_001 front; 013/014 back; `is_vfr` = interval spread > 5 ms, `pts_index.py` L96) | All timing comes from the PTS table, never from nominal fps; `nearest_frame` binary-searches true PTS. Frame counts from packets matched sequential decode counts for all 50 videos in Phase 12C forensics. | **HANDLED** |
| C2 | 29.97 fps back (pairs 015–018) consumed on a 60 Hz tick grid | `iter_synchronized_frames(tick_hz=60.0)` (L342) steps the reference clock; `VideoReader` single-frame cache (`video_reader.py` L52, L102–105) serves the repeated back frame at zero decode cost; status carries `is_new_back_frame=False`, `should_update_back_state=False`; `DeduplicatedTemporalGate` (L79) and the tracker's own guard (§5.3) prevent double state updates. | **HANDLED** (three independent layers, §7.5) |
| C3 | Duplicated/repeated presentation | Same mechanism as C2 — repetition is detected by **frame-index identity** (`idx != _last_frame_idx_*`, reader L270s), not by image comparison. Two distinct frames can never be merged; one frame can never be double-counted. | **HANDLED** |
| C4 | Duplicate PTS ticks inside one stream (two packets, same timestamp) | `FramePTSIndex` sorts ticks but does **not** deduplicate them; `frame_count` counts packets. `nearest_frame` would return one of the duplicates; decode order vs table order could diverge by the duplicate count. Not observed in this corpus (packet counts == decoded counts, 50/50). | **HANDLED-WITH-CAVEATS** for this dataset; **GAP-5 (LOW)** in general — Stage G validates `frame_count == decoded_count` per stream and fails the pair loudly if not (Stage H test S-DUP-PTS). |
| C5 | Dropped frames / timestamp gaps in a stream | Nothing special happens — and nothing needs to: a gap simply widens the local interval; `nearest_frame` returns the nearest surviving frame and Q5 grows accordingly (`local_frame_interval_ms` is reported per lookup). A *decode* failure mid-stream (`grab()` returns False) makes `get_frame_at_index` return `None` and the camera invalid for that tick (reader L93 ff.). | **HANDLED** for timing; decode-failure accounting is a pipeline duty (§7.11, ERR-DECODE) |
| C6 | Backward random access | `get_frame_at_index` with `frame_index < cursor` triggers a **full decoder reset and sequential re-grab from frame 0** (`video_reader.py` L108–110) — correct but O(n) per seek; the tick iterator is monotonic so this never fires in normal operation. The pipeline must never seek backwards (§7.4 invariant I3). | **HANDLED-WITH-CAVEATS** (performance trap, documented) |
| C7 | Ticks outside the overlap window | `get_synchronized_pair` still returns per-camera frames where they exist, with `is_in_overlap=False`, `valid_pair=False` (validity requires both frames **and** in-overlap). `get_frame_at_pts` returns `(None, None)` for targets outside `[−1e-4, duration + 1e-4]` and clamps inside that epsilon band (`video_reader.py` L135–138). Default `iter_synchronized_frames(overlap_only=True)` yields **only** `valid_pair` ticks — non-overlap content is invisible to consumers of the default iterator. | **HANDLED**, but note the asymmetry Stage G must respect: per-camera processing (Pass 1 pattern) covers full streams; tick iteration covers overlap only. The design (§7.4) processes full streams per camera and uses the overlap window only to scope association/diagnostics — matching how the GT was built. |
| C8 | Stream starts/ends inside the other's window (e.g. front shorter than back in 24/25 pairs) | Overlap end is tightened to true PTS durations in `__init__` (L167–176). Outside-overlap frames are still decodable per camera. | **HANDLED** |
| C9 | Missing camera file / missing pair | `SynchronizedVideoReader.__init__` constructs both `VideoReader`s unconditionally; a missing file raises `FileNotFoundError` (`video_reader.py` L25) and a missing pair id raises `ValueError` (reader L196) **at construction — there is no degraded single-camera mode**. Pass 1 had the same property (whole pair skipped if either video missing). | **GAP-2 (MEDIUM)** — Stage G treats this as a per-pair fatal error under failure isolation (§7.7): the pair is marked `FAILED_PRECONDITION`, the run continues with the other 24. A single-camera degraded mode is explicitly **not** designed (it would silently change association semantics); Stage H tests assert the failure behaviour (S-MISSING-CAM). |
| C10 | Metadata missing | Reader falls back to a second hard-coded metadata path before raising (L156–160). Behaviour is survivable but provenance-ambiguous. | **GAP-7 (LOW)** — pipeline pins the path in config (§7.10). |
| C11 | Non-monotonic tick requests / reuse of one reader for two passes | Dedup state (`_last_frame_idx_*`, `_tick_counter`) is mutable reader state; `iter_synchronized_frames` resets it, direct `get_synchronized_pair` calls do not. Out-of-order manual calls can corrupt `is_new_*` flags. | **HANDLED-WITH-CAVEATS** — Stage G uses one reader instance per pair per run, monotonic ticks only, or the per-camera loop (D2) which has no such state. |
| C12 | Drift within a pair | Not modelled at all (§3.8). | **GAP-1** (see §3.8) |

---

## 5. Audit D — Interface inventory (what Stage G orchestrates)

### 5.1 Phase 12C — synchronization

- `FramePTSIndex.from_video(path) -> FramePTSIndex` (`pts_index.py` L104): ffprobe packet-PTS
  table, OpenCV sequential fallback. Attributes: `pts_norm_sec[]`, `pts_raw_sec[]`,
  `frame_count`, `duration_norm_sec`, `effective_fps`, `is_vfr`, interval stats.
  `nearest_frame(target_norm_sec) -> PTSLookupResult{frame_index, actual_pts_norm_sec,
  actual_pts_raw_sec, delta_pts_ms, signed_delta_pts_ms, local_frame_interval_ms}` (L168).
- `VideoReader(path, use_pts_index=True)` (`video_reader.py` L23): `get_frame_at_index(idx)`,
  `get_frame_at_pts(t) -> (frame|None, PTSLookupResult|None)` (L129), `iter_frames(...)`,
  `get_metadata()`, `close()`; single-frame cache; counters `decode_call_count`,
  `cache_hit_count` (useful Stage J diagnostics).
- `SynchronizedVideoReader(pair_id, data_root="data/dual_camera_test_v1", metadata_dir=None,
  use_pts_index=True)`: `.sync_info: PairSynchronizationInfo{pair_id, front/back_path,
  initial/final/global_offset_sec, measured_drift_ms, epsilon_sync_ms, quality,
  overlap_start/end_back_sec, overlap_duration_sec, accuracy_classification}`;
  `.get_synchronized_pair(t_back_sec) -> (frame_back|None, frame_front|None, status: dict)`
  — status keys: `tick_index, t_ref_sec, t_back_target_sec, t_front_target_sec,
  t_back_actual_pts_sec, t_front_actual_pts_sec, t_back_raw_pts_sec, t_front_raw_pts_sec,
  frame_idx_back/front, delta_pts_back/front_ms, signed_*, inter_camera_pts_skew_ms,
  epsilon_sync_ms, global_offset_sec, valid_back/front_frame, is_new_back/front_frame,
  should_update_back/front_state, exposure_sync_model, is_in_overlap, valid_pair`;
  `.iter_synchronized_frames(step_frames=1, overlap_only=True, tick_hz=60.0)` (generator of
  `(frame_back, frame_front, status)`); `.create_side_by_side_preview(...)`; context manager.
- `DeduplicatedTemporalGate()` (L79): `.step_camera(camera: "front"|"back", is_new_frame,
  update_fn) -> (result, did_execute)`; counters per camera (`*_ticks_seen`,
  `*_unique_updates`, `*_skipped_duplicates`).

### 5.2 Phase 12D — detection

- `PerCameraDetector(camera_id, device=None, vehicle_conf_threshold=0.18,
  helmet_conf_threshold=0.25, vehicle_detector=None, helmet_detector=None)`
  (`camera_detector.py` L172). Constructor **validates** the Phase 9 class map equals
  `{0: no_helmet, 1: helmet}` and raises otherwise. `.detect_frame(frame_bgr, frame_index,
  pts_timestamp_sec) -> FrameDetectionBundle` (L208).
- `FrameDetectionBundle{camera_id, frame_index, pts_timestamp_sec, motorcycles[], persons[],
  helmet_boxes[], no_helmet_boxes[], all_heads[], vehicle/helmet/total_latency_ms, device}`
  of `StructuredDetection{..., category ∈ motorcycle|person|head, source_model ∈
  yolov8n_coco|phase9_best_helmet, camera_id, frame_index, pts_timestamp_sec}`; both have
  `to_dict()`. Semantic firewall (module docstring): COCO yields only motorcycle (3) /
  person (0); Phase 9 yields only cranial helmet classes; a head box is never a person.
- `SharedDetectorRegistry` (L134): process-level lazy singletons keyed by device — front and
  back detectors share one copy of each model in memory (the M1 8 GB provision, §7.6).
  Adapters (`VehiclePersonDetectorAdapter`, `HelmetDetectorAdapter`) resolve weights via the
  model registry; Stage I owns SHA-256 verification of `yolov8n.pt` (f59b3d83…) and Phase 9
  `best.pt` (0929a882…) before any detector is constructed.

### 5.3 Phase 12D tracking + embedded Phase 10 adapter

- `CameraTrackingPipeline(camera_id, coupler_config=None, bytetrack_config=None,
  phase10_config=None)`; `.from_yaml(camera_id, yaml_path="configs/tracking_config.yaml")`
  (L904) parses three YAML sections: `moto_rider_coupler`, `kalman_bytetrack`,
  `phase10_temporal_verification{state_machine, violation_decision}` (missing file → defaults).
- `.step(detection_bundle, should_update_state=True) -> dict` (L995). **State carried between
  frames (per camera instance):** `active_moto_tracks` / `terminated_moto_tracks`
  (`MotoTrack` → rider subtracks), `active_orphan_tracks` / `terminated_orphan_tracks`,
  ID counters, `frames_updated`, `duplicate_ticks_skipped`, `_last_processed_frame_index`,
  `_last_step_summary`. The step summary dict carries `active_moto_tracks`,
  `active_orphan_tracks`, `new_alerts`, `did_advance_state`.
- **Phase 10 inside the tracker:** each `RiderSubTrack` / `OrphanRiderTrack` owns a Phase 10
  `Track` (aggregator + `TemporalStateMachine`) and alerts come from `ViolationDecisionEngine`
  (`camera_tracker.py` imports, L22–24; `Phase10AdapterConfig` L43–51). This is the Phase 10
  surface the integrated system actually uses.

### 5.4 Phase 10 — the second surface (not in the integrated path)

`src/temporal/temporal_engine.py: TemporalVerificationEngine.process_frame(frame_index,
raw_detections: list[dict], fps=30.0)` is a standalone single-camera engine with its **own**
`RiderTracker`, aggregator, state machine, and decision engine, configured by
`TemporalEngineConfig` / `configs/temporal_config.yaml`. Neither `CameraTrackingPipeline` nor
Pass 1 uses it. **Design ruling:** the integrated pipeline uses §5.3 only. Running both would
double-count alerts and double-advance state machines on the same observations. The standalone
engine remains a Phase 10 unit-test surface (Stage H regression suite), nothing more (GAP-8,
documentation-level).

### 5.5 Phase 12E — association

- `CrossCameraAssociationEngine(config=None)`; `.from_yaml(yaml_path=
  "configs/cross_camera_association_config.yaml")` — sections: `appearance_features`,
  `tracklet_builder`, `transit_time_model`, `rider_cardinality`, `lane_spatial_context`,
  `convoy_order`, `association_scoring{cue_weights}`, `global_assignment`,
  `rider_association`, `orphan_association`; `{value: X, engineering_hypothesis: true}`
  wrappers unwrapped by `_unwrap_val`.
- `.associate_pipelines(pair_id, front_pipeline, back_pipeline, timestamp_quality,
  front_frames_by_index=None, back_frames_by_index=None, front_canonical_pts_by_frame=None,
  back_canonical_pts_by_frame=None, front_delta_pts_ms_by_frame=None,
  back_delta_pts_ms_by_frame=None, calibrated_homography_available=False)
  -> CrossCameraAssociationBatchResult` — the pipeline-facing entry point: it extracts
  tracklets from the two *finished* pipelines and runs candidate generation → multi-cue
  scoring → global 1-to-1 assignment → rider association → orphan association.
  `.associate_tracklets(...)` is the lower-level variant on pre-built tracklets.
- `CrossCameraAssociationBatchResult{...}.to_dict()` carries tracklets, candidate scores,
  `motorcycle_results`, `orphan_results`, `outcome_counts`, `rider_outcome_counts`,
  `orphan_outcome_counts`, `runtime_ms{...}`. Outcomes: LINKED / UNMATCHED /
  AMBIGUOUS_ABSTAIN / INSUFFICIENT_EVIDENCE (Stage K reports abstentions separately, never
  as correct).
- `TimestampQualityRecord{pair_id, reference_clock_anchor="back_stream_pts",
  global_offset_sec, epsilon_sync_ms, mean/max_delta_pts_front_ms, mean/max_delta_pts_back_ms,
  is_in_overlap}` (`tracklet_models.py` L43) — the synchronization→association contract.
- **Appearance features need frames:** `associate_pipelines` takes `frames_by_index` dicts;
  Pass 1 supplied ≤24 downscaled (1280×720) detection-bearing frames per camera (§7.6 adopts
  this bound as policy).

### 5.6 Config sources (complete list for Stage G)

`configs/tracking_config.yaml` (coupler + ByteTrack/Kalman + Phase 10 adapter),
`configs/cross_camera_association_config.yaml` (12E), `configs/temporal_config.yaml`
(standalone Phase 10 only — not consumed by the pipeline), Phase 11
`synchronization_metadata.json` (offset/residual/drift/quality per pair), model registry
(weights paths + SHA-256, consumed via adapters / Stage I). All config file hashes go into
the Stage J run manifest (§7.9).

---

## 6. Audit E — How orchestration actually happened in Pass 1 (evidence)

Pass 1 (Mission 1) is the only orchestration ever run over all 25 pairs, so its pattern is
evidence, not folklore. Verified in `work/annotate_pass1.py` + `work/run_pass1_all.py`:

1. **It never instantiated `SynchronizedVideoReader`.** Each camera was decoded independently
   and exhaustively: `cv2.VideoCapture` sequential `read()` until EOF, with timestamps taken
   from a separately built `FramePTSIndex` by frame ordinal (`pts_norm[n]`). One subprocess
   per pair (driver), front then back.
2. **Every physical frame was processed exactly once** — `detector.detect_frame(...)` then
   `pipe.step(bundle, should_update_state=True)` for every decoded frame. The new-frame
   problem never arose because there was no tick grid: repetition cannot occur when the loop
   is the decoder itself. This is the strongest empirical argument for D2/D3.
3. **Synchronization entered only at association time:** canonical maps
   `{i: pts_norm[i] + (offset if front else 0)}` were passed to
   `associate_pipelines(...)` together with a `TimestampQualityRecord` built from metadata
   (`offset_sec`, `residual_error_ms`) — plus the mislabelled nominal-grid "delta_pts" (§3.9).
4. **Checkpointing:** pair is done iff `<out>/pair_XXX/pair_XXX_summary.json` exists; the
   driver skips done pairs, logs per-pair stdout/stderr tails, records `failed[]` /
   `missing[]`, and always continues to the next pair — the failure-isolation and resume
   semantics §7.7 formalizes. Within a pair there was **no** mid-pair checkpoint: a crash at
   frame 900 of 1,000 restarts the pair (acceptable at ≤15 s clips; retained in the design).
5. **Memory precedent:** association frames were bounded by policy — only frames bearing
   motorcycle/head detections, every 3rd frame, capped at 24 per camera, downscaled to
   1280×720 (§7.6). The runner also called `gc.collect()` between cameras and tracked
   `peak_rss_mb` via `resource.getrusage` — both adopted in §7.6/§7.12.
6. **Missing-video semantics:** if either video was absent the pair returned early and was
   reported skipped — no partial outputs. §7.7 keeps this, upgraded to an explicit
   `FAILED_PRECONDITION` status (C9/GAP-2).
7. **Coverage accounting:** decoded count was checked against the container PTS count
   (`coverage_complete = (n == n_expected)`) per camera and recorded — §7.9 keeps this as a
   mandatory per-pair validation gate for Stage J.

Two orchestration patterns therefore exist in the project's history — tick-driven joint
iteration (12C reader demos/tests) and per-camera exhaustive decode + post-hoc canonical
mapping (Pass 1, the pattern that produced the frozen GT). They are **semantically
equivalent for detection/tracking** (per-camera state never depends on the other camera
mid-stream; coupling happens in 12E over finished tracklets), but they differ in coverage
(C7) and in which delta-PTS quantities exist (§3.9). The design picks the Pass 1 pattern as
primary precisely because the Stage K evaluation compares Stage J outputs against a GT built
that way.

---

## 7. Design — `src/pipeline/dual_camera_pipeline.py` (Stage G implementation spec)

### 7.1 Principles

P1. Orchestrate, never re-implement: no new synchronization math, no new tracker, no new
association logic. All time transforms are the §2.3 equations, executed by Phase 12C code or
by trivial canonical-map construction identical to Pass 1.
P2. Streams are consumed **forward-only, presentation-order, exactly-once** per physical frame.
P3. A pair is the unit of work, of checkpointing, and of failure.
P4. Every number emitted carries the §3 quantity name it actually is.
P5. Deterministic: same inputs + same config ⇒ byte-identical outputs (modulo timestamps in
the run manifest, which live outside per-pair artifacts).

### 7.2 Architecture

```
run_pipeline(pairs, config)                      # batch driver (one process per pair)
└── PairRun(pair_id, config)                     # per-pair orchestration
    ├── SyncContext          ← SynchronizedVideoReader.sync_info + FramePTSIndex ×2
    │                          (offset, epsilon, drift, quality, overlap window, PTS tables)
    ├── CameraRun("front")   ← VideoReader/PTS index → PerCameraDetector
    │                          → CameraTrackingPipeline.step (per physical frame)
    ├── CameraRun("back")    ← same, independent, sequential
    ├── AssociationRun       ← canonical PTS maps + TimestampQualityRecord
    │                          → CrossCameraAssociationEngine.associate_pipelines
    └── PairArtifacts        ← detections / tracks / association / events / sync diagnostics
                               + pair_summary.json (the checkpoint)
```

What the pipeline consumes from Phase 12C: `PairSynchronizationInfo` (offset, epsilon, drift,
quality, overlap), `FramePTSIndex` per stream, `VideoReader` decode + single-frame cache,
and — for diagnostics, previews, and Stage H — `SynchronizedVideoReader.get_synchronized_pair`
over the overlap window (sampled, §7.9) and `DeduplicatedTemporalGate` in tick mode (§7.5).
What it does **not** do: re-derive offsets, re-estimate residuals, or implement a second
nearest-frame lookup (only `FramePTSIndex.nearest_frame` exists in the design).

**Tick mode** (secondary, for Stage H and diagnostics): the same `PairRun` can iterate
`iter_synchronized_frames(tick_hz=60)` and route each camera through the gate (§7.5). It
exists to prove equivalence with the primary loop and to exercise C2/C3; Stage J uses the
primary loop.

### 7.3 API (design signatures — Stage G implements exactly these shapes)

```python
@dataclass(frozen=True)
class PipelineConfig:
    data_root: Path                 # dataset root (front/, back/, metadata/)
    metadata_path: Path             # pinned synchronization_metadata.json (GAP-7)
    tracking_config_path: Path      # configs/tracking_config.yaml
    association_config_path: Path   # configs/cross_camera_association_config.yaml
    output_root: Path               # Stage J artifact root
    device: str | None              # detector device; models via SharedDetectorRegistry
    vehicle_conf_threshold: float = 0.18
    helmet_conf_threshold: float = 0.25
    keyframe_max_per_camera: int = 24        # appearance-frame bound (§7.6)
    keyframe_stride: int = 3
    keyframe_max_width: int = 1280
    tick_hz: float = 60.0                    # tick mode / diagnostics only
    resume: bool = True

class PairStatus(Enum):
    PENDING / RUNNING / COMPLETE / FAILED_PRECONDITION / FAILED_RUNTIME / SKIPPED_COMPLETE

@dataclass
class PairRunResult:
    pair_id: str; status: PairStatus
    summary: dict | None            # parsed pair_summary.json when COMPLETE
    error: PairError | None         # §7.11 when failed

def run_pair(pair_id: str, config: PipelineConfig) -> PairRunResult
def run_all(pair_ids: list[str], config: PipelineConfig) -> list[PairRunResult]
    # sequential; per-pair subprocess isolation as in the Pass 1 driver (§7.7)
```

Internal (per pair): `SyncContext.from_reader(pair_id, config)`,
`run_camera(ctx, camera_id) -> CameraRunOutput{pipeline, coverage_rows, detection_jsonl_path,
track_frames_jsonl_path, frames_by_index, canonical_pts_by_frame, delta_pts_by_frame}`,
`run_association(ctx, front, back) -> batch_result`. Events for Stage J are **derived**
from the finished pipelines + batch result by a pure function
`derive_pair_events(...)` reusing the Phase 10 alerts already emitted inside tracker steps
(`new_alerts`) — the standalone `TemporalVerificationEngine` is never instantiated (§5.4).

### 7.4 Primary loop (per camera) — pseudocode and invariants

```
index = FramePTSIndex.from_video(video)            # Q1/Q2 tables
pipe  = CameraTrackingPipeline.from_yaml(cam, tracking_config)
det   = PerCameraDetector(cam, device)             # shared models via registry
for n, frame in enumerate(sequential_decode(video)):     # decoder order == PTS order
    assert n < index.frame_count
    bundle = det.detect_frame(frame, n, float(index.pts_norm_sec[n]))
    step   = pipe.step(bundle, should_update_state=True)  # n is new by construction
    stream coverage row + detections + step summary to JSONL (flush per frame)
canonical[cam] = {i: pts_norm[i] + (offset if cam == "front" else 0.0)}
```

Invariants: **I1** frame ordinal `n` is the presentation index (validated against
`index.frame_count` at end-of-stream; mismatch ⇒ ERR-COVERAGE, pair fails validation, §7.9).
**I2** `pts_timestamp_sec` passed downstream is always Q2 (normalized), never Q1.
**I3** no backward seeks, ever (C6). **I4** detection runs at most once per
`(camera, frame_index)` — enforced by loop construction and asserted via detector call
counters in Stage H. **I5** per-camera state is fully independent until `AssociationRun`.

### 7.5 The new-frame-only rule (Stage I requirement) — exact specification

**Definition.** A *genuinely new frame observation* for camera `c` at tick/loop step `k` is
one whose **presentation frame index** `idx_c(k)` (equivalently, whose normalized PTS
`pts_norm_c[idx]`, a strictly increasing key) differs from the index of the last observation
of camera `c` that was allowed to advance state. Repetition is an identity relation on
`(camera, frame_index)` — never on image content, never on wall-clock ticks.

**Enforcement points (defence in depth; all three already exist and are reused, not rebuilt):**

1. *Primary loop:* by construction (§7.4) — the decoder yields each physical frame once, so
   `is_new` is identically true and `should_update_state=True` is always correct.
2. *Tick mode:* `SynchronizedVideoReader` computes `is_new_*_frame` by index identity against
   `_last_frame_idx_*` and derives `should_update_*_state = is_new and valid`; the pipeline
   passes the bundle through `DeduplicatedTemporalGate.step_camera(cam, is_new, update_fn)`
   so `update_fn` (detect + step) executes only on new frames and the cached result is
   returned otherwise.
3. *Tracker backstop:* `CameraTrackingPipeline.step` independently refuses to advance when
   `should_update_state=False` **or** when `bundle.frame_index == _last_processed_frame_index`
   (`camera_tracker.py` L1006–1008), returning the cached summary with
   `did_advance_state=False` and incrementing `duplicate_ticks_skipped`.

Note the deliberate asymmetry the design preserves: in tick mode a repeated frame may still
be *displayed* (preview) and its cached detections *reported*, but detection is not re-run
(gate wraps detect+step together) and temporal state does not advance. A 30 fps frame
presented on two 60 Hz ticks therefore contributes exactly one detection pass and one
state-machine update — the precise Stage I requirement.

**Stage H verification hooks (§7.13):** S-NEW-1 (synthetic 60/30 pair on the tick grid:
back unique updates == back physical frames in overlap; front unique == ticks),
S-NEW-2 (frozen-frame injection: same index re-presented 5× ⇒ `duplicate_ticks_skipped`
increments, track state hash unchanged), S-NEW-3 (VFR burst: irregular intervals ⇒ updates
== distinct indices), S-NEW-4 (primary-loop equivalence: detections from tick mode ==
detections from primary loop on the same pair, by `(camera, frame_index)`).

### 7.6 Bounded memory (reference machine: Apple M1, 8 GB)

Budget per pair process (4K frames are 3840×2160×3 B ≈ 24.9 MB):

| Resident item | Bound |
|---|---|
| Decoded frames | ≤ 2 full-res (current per camera; `VideoReader` cache holds 1/stream) + tick-mode bundle 2 → **≤ 4 × 24.9 MB ≈ 100 MB** worst case |
| Appearance keyframes | ≤ 24/camera, ≤1280×720×3 B ≈ 2.8 MB each → **≤ 135 MB/pair**, released after association |
| Models | one shared instance per model via `SharedDetectorRegistry` (both cameras, one process) |
| PTS tables, track state, JSONL buffers | O(frames) small records; JSONL **streamed to disk per frame**, never accumulated |
| Previews/diagnostics | generated on downscaled copies only, never retained |

Hard rules: no whole-video frame lists; no `frames_by_index` beyond the keyframe policy;
readers closed (`close()` / context manager) and `gc.collect()` between cameras (Pass 1
precedent); the *batch* driver runs pairs in separate processes so a pair's memory is fully
reclaimed on exit; per-pair `peak_rss_mb` recorded in the summary (§7.12) so Stage J/L can
audit the bound empirically. Estimated steady-state RSS is dominated by the two YOLO models +
~250 MB of buffers — comfortably inside 8 GB with the OS and runtime.

### 7.7 Checkpointing, resume, failure isolation

- **Checkpoint = `pair_summary.json`**, written atomically (temp file + rename) as the last
  step of a successful pair, containing the §7.9 validation block. Existence + parse +
  `status == COMPLETE` ⇒ skip on resume (Pass 1 driver semantics, hardened: Pass 1 checked
  existence only; Stage G also validates the embedded input hashes, §7.8).
- **Partial artifacts** of a failed pair remain on disk under a `status` sidecar
  (`pair_status.json`: RUNNING/FAILED + error) and are never mistaken for complete outputs;
  re-running a pair overwrites its own directory only.
- **Failure isolation:** `run_all` executes each pair in a subprocess (Pass 1 pattern);
  a pair exception, OOM-kill, or precondition failure (§7.11) is recorded and the batch
  proceeds. Batch exit status summarizes counts; it is nonzero only if *zero* pairs complete
  or the run-level config itself is invalid.
- **Resume granularity is the pair.** Mid-pair resume is explicitly rejected: tracker and
  association state are not serializable without new machinery, and pairs are ≤ 15.1 s of
  video. (Recorded as accepted limitation L-1, §9.)

### 7.8 Determinism

Fixed iteration order (pair id ascending; front before back — Pass 1 order), no wall-clock
values inside per-pair artifacts (wall time lives in the batch log only), config + metadata +
weights SHA-256 recorded in `run_manifest.json`, JSON written with sorted keys where the
artifact is a summary/dictionary (JSONL row order = frame order by construction). Re-running
a completed corpus with `resume=False` must reproduce per-pair artifacts byte-identically
except `pair_summary.json`'s runtime fields — Stage H test S-DET-1 spot-checks this on
pair_001.

### 7.9 Stage J outputs (per pair, under `output_root/<pair_id>/`)

| Artifact | Content |
|---|---|
| `<pair>_front_coverage.csv`, `<pair>_back_coverage.csv` | per frame: `frame_index, pts_raw_sec (Q1), pts_norm_sec (Q2), canonical_pts_sec (§2.3), n_motorcycle_det, n_person_det, n_head_det, detector_ran, tracked counts` |
| `<pair>_<cam>_detections.jsonl` | per physical frame: `FrameDetectionBundle.to_dict()` |
| `<pair>_<cam>_track_frames.jsonl` | per physical frame: tracker step summary (active moto/orphan tracks, `new_alerts`, `did_advance_state`) |
| `<pair>_<cam>_tracks_final.json` | terminated + active track objects with PTS histories |
| `<pair>_association.json` | `CrossCameraAssociationBatchResult.to_dict()` |
| `<pair>_events.json` | derived events: per moto track — cross-camera link status (LINKED_BOTH_VIEWS / FRONT_VIEW_ONLY / BACK_VIEW_ONLY / AMBIGUOUS), rider states with provenance, Phase 10 alert references; plate fields `NOT_ASSESSED` (no OCR, per standing restrictions) |
| `<pair>_sync_diagnostics.json` | per pair: `offset_sec (Q3), epsilon_sync_ms (Q4), drift_ms (§3.8), quality`, per-camera PTS stats (`is_vfr`, interval min/median/max, effective fps), Q5 statistics **from true lookups in tick-mode diagnostic pass over the overlap** (mean/max `delta_pts_*`), `inter_camera_pts_skew_ms` distribution, overlap window + per-camera in-overlap frame counts, decode/cache counters, `exposure_sync_model` disclaimer (Q7) |
| `pair_summary.json` | checkpoint + validation block: frames expected (container) vs decoded per camera, `coverage_complete`, track/outcome counts, wall time, `peak_rss_mb`, input file SHA-256s, config SHA-256s |

Run-level: `run_manifest.json` (config, hashes, code version, per-pair status table) in
`output_root`. Validation gate per pair (all must pass before `COMPLETE`): coverage counts
match container counts; every detections row's `pts_timestamp_sec` equals the PTS table value;
association outcome counts reconcile with track inventories; sync diagnostics present with
all seven Q-fields populated.

### 7.10 Configuration surface

`PipelineConfig` (§7.3) is the only new config. Everything else is *referenced, not copied*:
tracking YAML, association YAML, metadata JSON (pinned path, GAP-7), model registry. The
pipeline records SHA-256 of each referenced file in the run manifest; a changed config
therefore invalidates resume-skips (summary stores the same hashes; mismatch ⇒ re-run pair).

### 7.11 Error taxonomy

| Code | Class | Examples | Pair outcome |
|---|---|---|---|
| ERR-PRECONDITION | inputs invalid before any processing | missing video/metadata pair (C9), pair id unknown, weights hash mismatch (Stage I), config file missing | `FAILED_PRECONDITION`, no artifacts |
| ERR-DECODE | stream decode failure | `grab()`/`read()` failure mid-stream, PTS table build failure | `FAILED_RUNTIME` unless partial coverage is *explicitly* accepted by Stage J policy (default: fail) |
| ERR-COVERAGE | decoded ≠ container frame count (C4/I1) | duplicate-PTS divergence, truncated file | pair fails validation — never marked COMPLETE |
| ERR-DETECTION | model/inference failure | detector exception, class-map validation failure (12D constructor) | `FAILED_RUNTIME` |
| ERR-TRACKING | tracker exception mid-pair | coupler/ByteTrack failure | `FAILED_RUNTIME` |
| ERR-ASSOCIATION | 12E failure | tracklet build/solve exception | `FAILED_RUNTIME` (per-camera artifacts retained, marked non-final) |
| ERR-RESOURCE | memory/time guard | RSS above configured ceiling | pair killed by driver, `FAILED_RUNTIME` with resource note |
| ERR-INTERNAL | anything else | — | `FAILED_RUNTIME`, full traceback in `pair_status.json` |

### 7.12 Logging / observability

Structured per-pair log lines (pair id, camera, frame milestones every 250 frames, phase
timings: decode+detect / tracking / association), gate counters in tick mode
(`unique_updates` vs `skipped_duplicates` per camera), `VideoReader` decode/cache counters
in sync diagnostics, driver log mirroring the Pass 1 format (`START/DONE/FAILED` + final
tally line), and `peak_rss_mb` per pair. No per-frame logging at INFO level (15,241 frames).

### 7.13 Stage H test hooks (design obligations)

Synthetic (generated streams with known PTS tables, via `FramePTSIndex` on synthetic files
or injected index objects): **S-6060** 60/60 aligned pair; **S-6030** 60/30 (C2/S-NEW-1);
**S-VFR** mixed VFR intervals incl. 013/014-style bursts; **S-DROP** dropped frames / PTS
gaps (C5); **S-DUP-PTS** duplicated timestamps (C4); **S-OFFSET-POS / S-OFFSET-NEG /
S-OFFSET-013** offset signs incl. the §2.2 worked example; **S-OVERLAP-EDGE** ticks exactly
at overlap start/end ± 1 frame (C7); **S-MISSING-CAM** (C9 behaviour); **S-NEW-1..4** (§7.5);
**S-DET-1** determinism (§7.8).
Real-video (Stage H list from the brief): pairs **001** (front VFR), **013** (large positive
offset + back VFR), **014** (back VFR), **015** (29.97 fps back + near-frame residual 37 ms),
**019** (MEDIUM quality, negative offset −2.380). Regression: existing suites
`tests/test_phase12c_pts_sync.py`, `tests/test_synchronized_reader.py`,
`tests/test_phase12d_tracking.py`, `tests/test_phase12e_association.py`,
`tests/temporal/*` must pass unweakened.

---

## 8. Consolidated GAP register

| ID | Gap | Severity | Owner / disposition |
|---|---|---|---|
| GAP-1 | Clock drift measured by Phase 11 (up to ±76 ms) is never applied or reported by the reader (§3.8) | HIGH (Stage K reporting) / MEDIUM (processing) | Stage G reports `drift_ms` per pair (§7.9); Stage K treats near-frame residuals as lower bounds; compensation is out of scope (D6, OQ-2) |
| GAP-2 | No degraded single-camera mode; missing camera = construction-time exception (C9) | MEDIUM | Stage G: `FAILED_PRECONDITION` + failure isolation (§7.7, §7.11); Stage H S-MISSING-CAM asserts behaviour |
| GAP-3 | Tick iterator's default mode hides all non-overlap content (C7) — any consumer assuming full-stream coverage from `iter_synchronized_frames` would silently lose frames | MEDIUM | Design D2: primary loop is per-camera full-stream; overlap scopes association/diagnostics only (§7.2, §7.4) |
| GAP-4 | "delta_pts" naming collision: Pass 1 stored nominal-grid deviation in Q5 fields consumed by 12E (§3.9) | MEDIUM | Stage G computes true Q5 in diagnostics and passes documented quantities in `TimestampQualityRecord`; field definitions recorded in Stage J outputs |
| GAP-5 | Duplicate PTS ticks not deduplicated in `FramePTSIndex` (C4) | LOW (not present in this corpus) | Stage G coverage validation (ERR-COVERAGE) detects divergence; Stage H S-DUP-PTS |
| GAP-6 | Reader `accuracy_classification` hard-codes a wrong band label and conflates residual with quality (§2.4) | LOW | Stage G emits `epsilon_sync_ms` and manifest `quality` as separate fields; label not reused |
| GAP-7 | Silent metadata fallback path in reader construction (C10) | LOW | `PipelineConfig.metadata_path` pins the file; hash in run manifest (§7.10) |
| GAP-8 | Two Phase 10 surfaces; standalone `TemporalVerificationEngine` could be double-wired (§5.4) | LOW (documentation) | Design ruling §5.4: pipeline uses the tracker-embedded adapter only |
| GAP-9 | Reader dedup state is mutable and order-sensitive for manual `get_synchronized_pair` callers (C11) | LOW | One reader per pair per run; monotonic ticks only (§7.2) |
| GAP-10 | Backward seeks are O(stream) decoder resets (C6) | LOW | Invariant I3: forward-only consumption (§7.4) |

**Contradictions with the mission brief, stated plainly:** (i) the brief's
`t_front = t_back − Δt_offset` is at best half the story — it is the *lookup* direction only,
and under the canonicalization reading it has the wrong sign; the code (§2) is authoritative
and needs no fix (§2.3). (ii) The brief's framing of Stage F as designing "the synchronization
engine" must not be read as licence to build a new one: the correct deliverable, implemented
in Stage G, is the orchestration layer of §7 around the existing Phase 12C engine. Everything
else in the brief (back = reference, offset from metadata, new-frame-only processing, bounded
memory, resume safety) was verified as consistent with the code.

## 9. Risks and open questions for Stages G–K

- **OQ-1 (Stage G):** Tick-mode/primary-loop equivalence (S-NEW-4) assumes detection is a pure
  function of the frame; GPU nondeterminism on Kaggle T4 could break byte-equality. Proposed
  resolution: compare by `(camera, frame_index)` with detection-set equality at fixed
  confidence thresholds on CPU for the equivalence test; Stage J itself always uses the
  primary loop, so production outputs are unaffected.
- **OQ-2 (Stage K, decision needed before L):** Should Stage K's synchronization report
  present drift-adjusted residual bounds (Q4 ± drift) for pairs 004/005/008/015/016/019?
  Proposed: yes, as a clearly-labelled bound, without changing any stored quantity.
  Carry-forward if unanswered: report Q4 and `drift_ms` side by side (the §7.9 default).
- **OQ-3 (Stage G/H):** `associate_pipelines` appearance features depend on which keyframes
  are supplied; Pass 1's policy (detection-bearing, stride 3, ≤24, 1280×720) is adopted
  verbatim for GT comparability. Any change requires re-validating association outcomes
  against the frozen release on the Stage H real-video pairs before Stage J.
- **OQ-4 (Stage J):** Venue — local VM (~1.5-core quota) vs Kaggle T4. Pass 1 took ~11 min of
  T4 for the full corpus; the integrated run adds per-frame tracking (already included) and
  diagnostics. Proposed: Kaggle T4 with the same dataset, outputs checksummed on download;
  never claim GPU if a pair ran on CPU (standing restriction).
- **OQ-5 (Stage K):** Event derivation — Stage J events derive from tracker alerts + 12E link
  status; the frozen GT's events derive from corrected track inventories. The matching rule
  (temporal overlap + track identity basis) must be pre-registered in Stage K before scoring,
  per the brief's "pre-defined temporal matching" requirement. Flagged now so it is not
  tuned post hoc.
- **L-1 (accepted limitation):** pair-granular resume only (§7.7). **L-2:** exposure
  synchronization is unprovable from container PTS (Q7) and bounds every synchronization
  claim in Stages J–L. **L-3:** MEDIUM-quality pairs (008/019/022) and near-frame pairs
  (004/005/008/015) carry irreducibly larger alignment uncertainty; Stage K must report
  their metrics stratified, not only pooled.

## Appendix A — Per-pair synchronization table (Phase 11 `pair_manifest.csv`, verified)

| Pair | offset_sec (Q3) | residual ε_sync ms (Q4) | drift_ms | quality | overlap s | front dur s | back dur s |
|---|---|---|---|---|---|---|---|
| pair_001 | −0.205 | 3 | +0 | HIGH | 3.102 | 3.31 | 6.63 |
| pair_002 | −1.055 | 3 | +2 | HIGH | 4.214 | 5.27 | 7.06 |
| pair_003 | +0.665 | 3 | +0 | HIGH | 3.947 | 3.95 | 7.27 |
| pair_004 | −1.285 | 18 | +36 | HIGH | 1.766 | 3.05 | 3.37 |
| pair_005 | +0.540 | 35 | +70 | HIGH | 3.477 | 3.48 | 5.89 |
| pair_006 | −0.045 | 6 | −12 | HIGH | 2.238 | 2.28 | 4.26 |
| pair_007 | +0.415 | 4 | −8 | HIGH | 2.645 | 2.65 | 3.99 |
| pair_008 | +0.755 | 38 | −76 | MEDIUM | 2.645 | 2.65 | 4.52 |
| pair_009 | +0.225 | 3 | +0 | HIGH | 3.904 | 3.90 | 6.93 |
| pair_010 | +0.080 | 7 | +14 | HIGH | 1.899 | 1.90 | 4.29 |
| pair_011 | −0.725 | 3 | +0 | HIGH | 3.350 | 4.08 | 4.61 |
| pair_012 | +1.395 | 3 | +0 | HIGH | 4.032 | 4.03 | 7.44 |
| pair_013 | +6.695 | 3 | +0 | HIGH | 6.784 | 6.78 | 15.14 |
| pair_014 | −1.820 | 3 | +0 | HIGH | 2.660 | 4.48 | 5.70 |
| pair_015 | −1.470 | 37 | +74 | HIGH | 4.247 | 5.72 | 6.36 |
| pair_016 | −1.675 | 10 | −20 | HIGH | 7.584 | 9.26 | 10.35 |
| pair_017 | +0.355 | 3 | +0 | HIGH | 3.392 | 3.39 | 6.68 |
| pair_018 | +1.405 | 3 | +0 | HIGH | 2.624 | 2.62 | 6.61 |
| pair_019 | −2.380 | 8 | −16 | MEDIUM | 3.860 | 6.27 | 3.86 |
| pair_020 | +3.320 | 3 | +0 | HIGH | 2.688 | 2.69 | 8.19 |
| pair_021 | −1.250 | 3 | +0 | HIGH | 4.574 | 5.82 | 6.40 |
| pair_022 | −1.565 | 3 | +0 | MEDIUM | 3.683 | 5.25 | 5.54 |
| pair_023 | −0.030 | 3 | +0 | HIGH | 5.367 | 5.40 | 7.19 |
| pair_024 | +1.235 | 3 | +0 | HIGH | 8.043 | 8.04 | 12.29 |
| pair_025 | −0.590 | 3 | +0 | HIGH | 3.527 | 4.12 | 5.61 |

Notes: back streams are 59.94 fps (pairs 001–012, 019–025), VFR (013, 014), 29.97 fps
(015–018); front streams ~60 fps with pair_001 front forensically VFR. `offset_sec` here is
identical in `synchronization_metadata.json` and `pair_manifest.csv` for all 25 pairs.

## Appendix B — Evidence index

Sign conventions: `synchronized_reader.py` L133, L198, L208–209, L246 (target formula in
`get_synchronized_pair`), L302, L167–176. PTS: `pts_index.py` L58–59, L72, L96, L104, L168.
Decode/cache: `video_reader.py` L52, L58, L93, L102–110, L129, L135–138. Gate:
`synchronized_reader.py` L79–124. Detector: `camera_detector.py` L134, L172, L208. Tracker:
`camera_tracker.py` L43–51, L866, L904, L995, L1006–1008. Association:
`cross_camera_association.py` L91, L122, L293, L393; `tracklet_models.py` L43–56;
`transit_time_model.py` L37–39, L165 ff. Pass 1: `work/annotate_pass1.py` (canonical map,
`TimestampQualityRecord` construction, keyframe policy), `work/run_pass1_all.py`
(checkpoint/driver semantics).

*End of Stage F design. Next: Stage G — implement `src/pipeline/dual_camera_pipeline.py` to
this specification.*
