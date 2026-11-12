# TASK-04 Implementation Report — Genuine-Observation Helmet Track-State Export (GAP-A)

Baseline: `dd7ab7348433bf914c14e880464919aa56b1ac4d` (verified local == origin/main before changes). Design: `ARGON_TASK03_HELMET_TRACK_STATE_EXPORT_DESIGN.md`.

## 1. Root cause addressed
- `EMAAggregator` returns zero mass once its 10-entry history holds no `is_detected` entry; `RiderSubTrack`s are stepped on every parent frame, so 73/85 subtracks flushed to `{0,0}` at export (orphans terminate at 9 unmatched steps and never flush). *Online EMA left untouched.*
- `run_camera()` counted `subtrack_state_counts` on every active moto frame, including non-coupled ones (stale votes).

## 2. Architecture implemented
`HelmetEvidenceAccumulator` (O(1) memory) in `src/tracking/camera_tracker.py`, owned by `RiderSubTrack` and `OrphanRiderTrack`. It is written **only** from the genuine matched path (`__init__` initial observation and `update_matched`); `update_unmatched`/Kalman prediction never touch it. Repeated or non-increasing frame indices are ignored; non-helmet classes are ignored. Online `Track`/`EMAAggregator`/`ViolationStateMachine`, `aggregated_confidence`, `alerts_emitted`, `temporal_state` are unchanged.

## 3. Files changed
- `src/tracking/camera_tracker.py` (accumulator, derive helpers, exports, `ema_alpha` forwarded)
- `src/temporal/rider_tracker.py` (`Track.__init__(ema_alpha=0.35)` -> `create_aggregator(alpha=...)`)
- `src/pipeline/dual_camera_pipeline.py` (`run_camera` counts only `time_since_update == 0`; `_track_records` uses `sub.track_helmet_state`)
- `scripts/reproducibility/track_state_redesign.py` (new amended evaluation, `pts_history` fallback), `event_regression_task04.py` (new), small additive hooks in `extract.py`/`evaluate.py` (`track_state_redesign`, `helmet_track_level_redesign`; legacy blocks untouched)
- `tests/test_phase12_track_state_export.py` (16 tests), roadmap status, this report, the design doc.

## 4. Export schema (both track types)
`track_helmet_state` (= strict majority), `_majority`, `_conf_weighted`, `_last_coupled` (EMA snapshot at last genuine observation, tie -> UNCERTAIN), `_ema_final` (legacy Stage K rule, tie -> HELMET), `last_coupled_aggregated_confidence`, `coupled_confidence_weights`, `coupled_observations_count`, `coupled_helmet_count`, `coupled_no_helmet_count`, `coupled_state_switches`, `last_coupled_frame_index`, `last_coupled_pts_sec`. Majority tie / zero observations -> `UNCERTAIN`; a single observation decides (no K>=2 gate). Existing keys preserved. No schema deviation from the design.

## 5. Tests
New: 16/16 pass (A–I plus ties, single observation, missing timestamp, serialization, orphan/subtrack parity, `ema_alpha` forwarding, repeated-update protection). Full suite: see section 9.

## 6. Event-level regression (Stage J, no inference)
`results/phase12_track_state_redesign/event_regression.json`: **85 events compared, 0 changed** (rider states/provenance, `violation_event_type`, `event_is_violation`). **Discrepancy:** the design/prompt cite 67 motorcycle events; the artifacts contain 85 events in total (67 is presumably the associated subset). All 85 are unchanged, so the 67 are as well. Alert ids derive from unchanged online state; windows were not recomputed (depend only on unchanged states/track times).

## 7. Amended evaluation (`results/phase12_track_state_redesign/`, 209 decided GT tracks, source = `pts_history` fallback)
| Method | Decided | Abstain | Coverage | Cond. acc. | End-to-end |
|---|---|---|---|---|---|
| ema_final (legacy, reproduces Stage K) | 136 | 73 (window flush) | 0.651 | 0.985 | 0.641 |
| last_coupled | 209 | 0 | 1.000 | 0.986 | 0.986 |
| majority = **primary** | 209 | 0 | 1.000 | 0.990 | 0.990 |
| conf_weighted | 209 | 0 | 1.000 | 0.990 | 0.990 |

Primary confusion: HELMET→HELMET 56, HELMET→NO_HELMET 1, NO_HELMET→HELMET 1, NO_HELMET→NO_HELMET 151. Subgroups (full detail in JSON): K=1 85/85 correct; K>=2 122/124; moto subtracks 84/85; orphans 123/124; legacy-flushed (tail-gap proxy) 72/73; tracks with ≥1 state switch 17/19 vs 190/190 without switches. Single `occluded_unknown` GT track is outside the 209. These numbers are **a re-score of existing Stage J artifacts, not a new inference run**; thresholds were not tuned. They are computed on the same GT used to design the method, so they are not independent validation.

## 8. Performance / memory
Accumulator: ~10 scalars/track, no history retained (existing `pts_history` untouched). Evaluation reads JSON only; no models, no video, no GPU. 6144 MiB ceiling unchanged.

## 9. Limitations
- Tail-gap subgroup uses legacy-window-flush as a proxy (`track_frames.jsonl` not in repo).
- Existing `extract.py`/`evaluate.py` still reference a different machine's `~/workspace` paths; the redesign hooks are additive and the standalone runner uses repo paths.
- Majority can still be wrong on low-confidence or manually-corrected tracks (2 of 209); the historical majority differs from the latest state on transition tracks, both exported.

## 10. Commit / push
Implementation commit `81bc3e60f9c82770190e192b1d252b042129d4c9` pushed to `origin/main`; remote HEAD verified equal to local HEAD at that time.
