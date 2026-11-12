# Integrated Dual-Camera Helmet System — SYNCHRONIZATION REPORT (Stage L)

**Status: COMPLETE (2026-11-09).** All values are from the Stage J run's
per-pair `sync_diagnostics.json`, consolidated in Stage K's
`work/stageK/sync_by_pair.csv` (25 rows) and re-verified in Stage L (strata
medians recomputed from the CSV match the Stage K report exactly). Stage J gates
verified decoded frame counts == Phase 12C forensic counts for all 50 videos
(front 6,557 / back 8,684 = 15,241).

## 1. Conventions (Stage F audit, code-authoritative)

- **Back camera = reference timeline.** The operative offset is the Phase 11
  metadata `offset_sec` (= pair_manifest `temporal_offset_sec`) for all 25 pairs.
- Canonical mapping actually consumed by association and event derivation:
  **`t_ref = front_pts_norm + offset_sec`** (back frames: `t_ref = pts_norm`).
  The reader's lookup runs the other direction (`t_front_target = t_ref −
  offset_sec`, `synchronized_reader.py` L246); the Stage F audit confirmed there
  is **no sign error in the code** — the brief's prose formula described only the
  lookup direction.
- Seven time quantities are kept distinct everywhere in the system and in this
  report: **Q1** raw container PTS · **Q2** normalized PTS · **Q3** clock offset
  (`offset_sec`) · **Q4** ε_sync (Phase 11 residual after offset alignment) ·
  **Q5** per-lookup delta-PTS (true nearest-available-frame error, computed by
  the pipeline — *not* Pass 1's stored "delta_pts", which the Stage F audit
  showed was nominal-grid deviation, GAP-4) · **Q6** cross-camera transit delay
  (a scene property used by Phase 12E association, never a clock correction) ·
  **Q7** exposure synchronization.
- **Exposure-sync disclaimer (mandatory):** nothing in this stack establishes
  physical exposure synchronization. The two cameras are independent consumer
  devices (iPhone front, iPad back) with no genlock; alignment is at container
  timestamp level only. No statement in this report or in the system output
  should be read as claiming the two sensors exposed corresponding frames at the
  same physical instant.

## 2. Residual summary (Stage K §9, recomputed and confirmed in Stage L)

| Stratum | Pairs | Median ε_sync | Median \|drift\| | Max \|drift\| | Median Q5 skew max |
|---|---|---|---|---|---|
| Pooled | 25 | **3 ms** | 0 ms | 76 ms | 7.8 ms |
| Near-frame (004/005/008/015) | 4 | **36 ms** | **72 ms** | 76 ms | 5.4 ms |
| MEDIUM quality (008/019/022) | 3 | 8 ms | 16 ms | 76 ms | 3.5 ms |

Sync quality profile (Phase 11): HIGH ×22, MEDIUM ×3 (pairs 008, 019, 022).
For 21 of 25 pairs the residual ε_sync is sub-frame (3 ms nominal); the four
near-frame pairs (004, 005, 008, 015; residuals 18–38 ms) are the only ones whose
alignment uncertainty approaches a frame interval.

## 3. Per-pair table (Stage J diagnostics)

| Pair | Offset (s) | ε_sync (ms) | Drift (ms) | Quality | Q5 skew max (ms) | Labelled bound ε+\|drift\| (ms) | Decoded F / B |
|---|---|---|---|---|---|---|---|
| pair_001 | -0.205 | 3 | +0 | HIGH | 11.5 | — | 206 / 397 |
| pair_002 | -1.055 | 3 | +2 | HIGH | 11.4 | — | 313 / 423 |
| pair_003 | +0.665 | 3 | +0 | HIGH | 7.6 | — | 234 / 436 |
| pair_004 | -1.285 | 18 | +36 | HIGH | 4.1 | 54.0 | 179 / 201 |
| pair_005 | +0.540 | 35 | +70 | HIGH | 6.7 | 105.0 | 206 / 352 |
| pair_006 | -0.045 | 6 | -12 | HIGH | 5.0 | — | 135 / 255 |
| pair_007 | +0.415 | 4 | -8 | HIGH | 5.6 | — | 157 / 239 |
| pair_008 | +0.755 | 38 | -76 | MEDIUM | 3.5 | 114.0 | 156 / 270 |
| pair_009 | +0.225 | 3 | +0 | HIGH | 8.4 | — | 232 / 415 |
| pair_010 | +0.080 | 7 | +14 | HIGH | 5.6 | — | 110 / 256 |
| pair_011 | -0.725 | 3 | +0 | HIGH | 8.1 | — | 241 / 275 |
| pair_012 | +1.395 | 3 | +0 | HIGH | 8.6 | — | 238 / 446 |
| pair_013 | +6.695 | 3 | +0 | HIGH | 17.7 | — | 404 / 608 |
| pair_014 | -1.820 | 3 | +0 | HIGH | 7.8 | — | 266 / 276 |
| pair_015 | -1.470 | 37 | +74 | HIGH | 20.5 | 111.0 | 339 / 190 |
| pair_016 | -1.675 | 10 | -20 | HIGH | 24.7 | 30.0 | 551 / 310 |
| pair_017 | +0.355 | 3 | +0 | HIGH | 17.5 | — | 200 / 200 |
| pair_018 | +1.405 | 3 | +0 | HIGH | 17.5 | — | 155 / 198 |
| pair_019 | -2.380 | 8 | -16 | MEDIUM | 3.3 | 24.0 | 372 / 231 |
| pair_020 | +3.320 | 3 | +0 | HIGH | 4.0 | — | 158 / 490 |
| pair_021 | -1.250 | 3 | +0 | HIGH | 6.5 | — | 348 / 383 |
| pair_022 | -1.565 | 3 | +0 | MEDIUM | 4.6 | — | 312 / 332 |
| pair_023 | -0.030 | 3 | +0 | HIGH | 4.7 | — | 320 / 430 |
| pair_024 | +1.235 | 3 | +0 | HIGH | 10.1 | — | 481 / 736 |
| pair_025 | -0.590 | 3 | +0 | HIGH | 11.0 | — | 244 / 335 |

Offsets range from −2.380 s (pair_019) to **+6.695 s (pair_013)** — the large
pair_013 offset is real (late front-camera start) and is handled exactly by the
constant-offset mapping; Stage H's S-OFFSET-013 test pins that the operative
value is `offset_sec`, not the metadata's unused `initial/final_offset_sec`
(pair_013: +8.357 / +4.595, loaded but never consumed).

## 4. GAP-1 — clock drift is reported, never applied (the dominant uncertainty)

Phase 11 measured per-pair clock drift, and four pairs carry substantial values:
**pair_004 +36 ms, pair_005 +70 ms, pair_008 −76 ms, pair_015 +74 ms** (further
non-zero drift: 002 +2, 006 −12, 007 −8, 010 +14, 016 −20, 019 −16 ms). The
synchronization reader applies only the constant offset; **no component of the
system compensates drift** (Stage F decision D6: report, never compensate —
inventing a drift correction without a validated clock model would falsify the
record).

Consequences, stated as bounds rather than hidden:

- Where drift ≠ 0, the stored ε_sync is a **lower bound** on end-of-window
  misalignment: a pair's alignment error grows from ε_sync at the alignment
  anchor toward ε_sync + |drift| across the window.
- Following Stage F design OQ-2, the labelled bound ε_sync + |drift| is reported
  alongside (never instead of) the raw values for the six pairs where it matters
  (004: 54 ms; 005: 105 ms; 008: 114 ms; 015: 111 ms; 016: 30 ms; 019: 24 ms) —
  see the table above and `work/stageK/sync_by_pair.csv`.
- Stage K's event timing errors (0.0 s on the reference timeline) inherit this
  caveat: on drift-affected pairs, wall-clock timing carries up to ±76 ms of
  additional bounded uncertainty.
- All near-frame residuals in this report and in the evaluation must be read as
  **lower bounds**.

## 5. Q5 — nearest-frame lookup error stays small

The pipeline computes the true per-lookup delta-PTS (Q5) over each pair's overlap
(Stage J diagnostics; Stage K consolidation). Pooled median Q5 skew max is
7.8 ms; even on the near-frame pairs the median Q5 skew max is 5.4 ms, and the
worst single-pair Q5 skew max is 24.7 ms (pair_016). Frame *lookup* is therefore
not the system's synchronization problem — unapplied clock *drift* is (§4).
This distinction was only possible after the Stage F audit separated Q5 from
Pass 1's mislabelled nominal-grid deviation (GAP-4): on VFR pair_013's back
stream that grid deviation reaches 2,531 ms while its true Q5 max is 17.7 ms —
quoting the former as a sync error would have been wrong by two orders of
magnitude.

## 6. New-frame-only behaviour (Stage I requirement, Stage H/J evidence)

A camera's detector/tracker state is updated only on a genuinely new frame
observation, keyed on (camera, frame_index) identity — never re-updating a
repeated 30 FPS frame on 60 FPS ticks. Evidence: Stage H tick-mode tests
(59 ticks → back 30 unique updates / 29 gated duplicates; frozen-frame injection
leaves tracker state hash unchanged); Stage I instrumented real run (detector
calls == decoded frames == coverage rows, both cameras); Stage J gates (JSONL
lines == frames for all 50 streams).

## 7. Test evidence

Stage H synchronization coverage: synthetic battery 16/16 (60/60 and 60/30
pairings, VFR mixes, dropped frames, duplicate PTS, positive/negative/013
offsets, overlap boundaries, missing camera with batch isolation) and semantics
14/14 (canonical mapping asserted on artifacts, GAP-1 drift reported-but-never-
applied, GAP-4 numeric separation, resume lifecycle, all 8 error codes);
real-video battery 24/24 on pairs 001/013/014/015/019 with PTS counts exactly
equal to the Phase 12C forensic counts on all 10 streams and three-way offset
agreement (metadata == manifest == reader, exact). Full detail:
`repo/docs/phase12/POST_ANNOTATION_STAGE_H_TEST_REPORT.md`.

## 8. Remaining synchronization gaps (from the Stage F audit, dispositioned)

- **GAP-1 (HIGH)** — drift never applied: accepted as a documented limitation
  with labelled bounds (§4); compensation would require a validated clock model
  and remains future work, not a silent fix.
- **GAP-2 (MEDIUM)** — no degraded single-camera mode: a missing camera fails
  the pair with `FAILED_PRECONDITION` and the batch continues (by design;
  Stage H S-MISSING-CAM verified).
- **GAP-3 (MEDIUM)** — the tick iterator yields overlap-only content: retained
  for diagnostics; the primary loop is per-camera exhaustive decode, so no
  annotated content is hidden from the pipeline.
- **GAP-5..10 (LOW)** — duplicate PTS ticks not deduplicated upstream (absent in
  this corpus; coverage validation would catch it), reader accuracy-band
  labelling, silent metadata fallback, mutable dedup state, forward-only seek
  invariant, and the two Phase 10 surfaces (the pipeline uses only the
  tracker-embedded adapter) — all recorded in the Stage F design document with
  dispositions; none affects the numbers in this report.
