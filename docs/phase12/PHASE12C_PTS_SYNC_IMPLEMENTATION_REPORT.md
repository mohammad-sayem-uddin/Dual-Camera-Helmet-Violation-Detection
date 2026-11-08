# PHASE 12C — PTS-INDEXED SYNCHRONIZED DUAL-CAMERA FRAME READER IMPLEMENTATION REPORT

**Project:** An Integrated Dual-Camera Framework for Context-Aware Motorcycle Helmet Violation Detection with Confidence-Aware Temporal Verification  
**Repository:** `https://github.com/mohammad-sayem-uddin/Dual-Camera-Helmet-Violation-Detection` (`main` branch)  
**Architect & Lead Engineer:** Gemini 4 Argon  
**Scope:** Implementation and verification of the PTS-Indexed Synchronized Dual-Camera Frame Reader (`FramePTSIndex`, `VideoReader`, `SynchronizedVideoReader`, and `DeduplicatedTemporalGate`).

---

## 1. Executive Summary

Phase 12C resolves three critical container-timing hazards identified in the 25-pair dual-camera benchmark dataset (`data/dual_camera_test_v1/`):
1. **Variable Frame Rate (VFR) Dynamic Throttling**: `pair_013_back.mp4` (`40.15` effective fps) and `pair_014_back.mp4` (`48.47` effective fps)—plus a single dropped-frame interval in `pair_001_front.mp4`—cannot be indexed via nominal `round(t * avg_fps)` or sought via OpenCV's `cv2.CAP_PROP_POS_FRAMES`, which internally converts frame numbers to timestamps using average FPS and lands up to **`2,870.00 ms` (2.87 seconds / 86 frames)** off target on `pair_013_back.mp4`.
2. **Mixed 60 fps / 30 fps Asymmetry**: `pair_015_back.mp4` through `pair_018_back.mp4` are 30 fps CFR streams paired with 60 fps CFR Front streams, while `pair_013_back.mp4` runs at 30 fps across its entire common overlap window (`[6.695 s, 13.410 s]`). Stepping at a canonical 60 Hz reference timeline without deduplication would run the Back-camera detector twice on every 30 fps frame and prematurely trigger Phase 10 temporal state-machine confirmations in half the required physical frames.
3. **Metrological Separation of Uncertainty Terms**: Container PTS discretization offsets ($\delta_{\text{pts},\text{front}}, \delta_{\text{pts},\text{back}}$) are strictly separated from Phase 11.6 inter-camera synchronization residual uncertainty ($\epsilon_{\text{sync}}$), and container timestamp alignment is explicitly distinguished from physical CMOS rolling-shutter exposure synchronization.

---

## 2. Implemented & Upgraded Modules

| File Path | Role & Key Capabilities |
| :--- | :--- |
| [`src/synchronization/pts_index.py`](file:///Users/sayemuddin/Desktop/research_helmet/src/synchronization/pts_index.py) | Implements [`FramePTSIndex`](file:///Users/sayemuddin/Desktop/research_helmet/src/synchronization/pts_index.py#L34-L250), [`PTSLookupResult`](file:///Users/sayemuddin/Desktop/research_helmet/src/synchronization/pts_index.py#L22-L31), and [`validate_pts_against_opencv_sequential`](file:///Users/sayemuddin/Desktop/research_helmet/src/synchronization/pts_index.py#L253-L299). Extracts exact rational `time_base` and packet `pts` ticks via `ffprobe` sorted into presentation order, normalizes Frame 0 to `0.000000 s`, performs $O(\log N)$ nearest-frame lookup via `np.searchsorted`, and computes exact discretization residuals (`delta_pts_ms`, `signed_delta_pts_ms`). |
| [`src/video/video_reader.py`](file:///Users/sayemuddin/Desktop/research_helmet/src/video/video_reader.py) | Upgrades [`VideoReader`](file:///Users/sayemuddin/Desktop/research_helmet/src/video/video_reader.py#L16-L220) to use `FramePTSIndex`, replaces broken `cv2.CAP_PROP_POS_FRAMES` seeking with a **Sequential Forward Decoder Cursor (`_cursor_idx`)** using `cap.grab()`/`cap.retrieve()`, and adds a single-frame cache (`_cached_frame_idx`) so repeated 30 fps frame queries during 60 Hz iteration return in `0.0 ms` without re-decoding. |
| [`src/synchronization/synchronized_reader.py`](file:///Users/sayemuddin/Desktop/research_helmet/src/synchronization/synchronized_reader.py) | Upgrades [`SynchronizedVideoReader`](file:///Users/sayemuddin/Desktop/research_helmet/src/synchronization/synchronized_reader.py#L106-L356), adds [`SyncedFramePair`](file:///Users/sayemuddin/Desktop/research_helmet/src/synchronization/synchronized_reader.py#L43-L69), and introduces [`DeduplicatedTemporalGate`](file:///Users/sayemuddin/Desktop/research_helmet/src/synchronization/synchronized_reader.py#L72-L103). Enforces strict overlap validity (`valid_pair = bool(valid_back and valid_front and in_overlap)`), decouples per-camera update eligibility (`should_update_back_state = bool(is_new_b and valid_back)`, `should_update_front_state = bool(is_new_f and valid_front)`), and separates `epsilon_sync_ms`, `delta_pts_front_ms`, `delta_pts_back_ms`, and `inter_camera_pts_skew_ms`. |
| [`tests/test_phase12c_pts_sync.py`](file:///Users/sayemuddin/Desktop/research_helmet/tests/test_phase12c_pts_sync.py) | 7 unit and forensic integration tests verifying VFR PTS indexing, `CAP_PROP_POS_MSEC` validation, `CAP_PROP_POS_FRAMES` bug prevention, uncertainty separation, non-zero discretization residuals, production-path 30 fps duplicate-frame gating (`get_synchronized_pair()` -> `status["should_update_back_state"]` -> `DeduplicatedTemporalGate` -> Phase 10 `TemporalStateMachine`), and decoupled single-camera update eligibility outside overlap. |
| [`scripts/phase12c_forensic_pts_audit.py`](file:///Users/sayemuddin/Desktop/research_helmet/scripts/phase12c_forensic_pts_audit.py) | Reproducible audit script that validates all 50 video containers and all 25 synchronized pairs, exporting CSV and JSON forensic artifacts to `results/phase12c_pts_sync/`. |

---

## 3. Verification of Mandatory Conditions

### Condition 1: Validation of `CAP_PROP_POS_MSEC` Against Decoded Frame Timestamps
We did **not** assume `CAP_PROP_POS_MSEC` provides exact PTS without verification. Across all 50 video streams (`25 front + 25 back`), [`validate_pts_against_opencv_sequential`](file:///Users/sayemuddin/Desktop/research_helmet/src/synchronization/pts_index.py#L253-L299) compared `ffprobe` rational packet PTS (`pts_ticks * (time_base_num / time_base_den)`) against OpenCV `CAP_PROP_POS_MSEC` under two access patterns:
- **Sequential Forward Decoding (`cap.grab()`)**: Across all 50 videos (15,637 total frames), sequential `CAP_PROP_POS_MSEC` matches stream-normalized rational PTS `(pts[k] - pts[0]) * time_base` to `0.000000 ms` (`max_seq_pos_msec_vs_norm_pts_diff_ms = 0.000000 ms`), while raw unnormalized `ffprobe` `start_time` differs by `31.3 ms` to `79.8 ms` on 24/25 Android Front streams due to an audio-stream lead before the first video packet.
- **Random Frame Seeking (`cap.set(cv2.CAP_PROP_POS_FRAMES, k)`)**: OpenCV's FFmpeg backend fails severely on VFR streams when `CAP_PROP_POS_FRAMES` is used, landing **`2,870.00 ms`** off at midpoint frame `304` on `pair_013_back.mp4`, **`350.00 ms`** off at midpoint frame `138` on `pair_014_back.mp4`, and **`16.63 ms`** off on `pair_001_front.mp4`. Consequently, `VideoReader` bans `CAP_PROP_POS_FRAMES` and uses sequential forward `grab()` cursor advancement.

### Condition 2 & Condition 3: Four-Way Metrological Distinction & Overlap Validity
Every status dictionary returned by `SynchronizedVideoReader.get_synchronized_pair()` and `iter_synchronized_frames()` strictly distinguishes four separate timing concepts:
1. **Container PTS Agreement**: `CAP_PROP_POS_MSEC` (under sequential `grab()`) vs. stream-normalized `ffprobe` rational packet PTS ($0.000000\text{ ms}$ difference across all 50 streams).
2. **Frame-Selection Discretization Residuals (`delta_pts_front_ms`, `delta_pts_back_ms`)**: $|t_{c,\text{actual\_pts}} - t_{c,\text{target}}| \times 1000\text{ ms}$, plus `inter_camera_pts_skew_ms` ($|(t_{B,\text{actual\_pts}} - t_{F,\text{actual\_pts}}) - \Delta t_{\text{rec}}| \times 1000\text{ ms}$).
3. **Phase 11.6 Inter-Camera Synchronization Residual (`epsilon_sync_ms`)**: Empirical synchronization uncertainty ($1.0\text{--}10.0\text{ ms}$ on the 21 sub-frame pairs; $18.0\text{--}38.0\text{ ms}$ on the 4 near-frame pairs).
4. **Physical Exposure Synchronization Disclaimer**: `exposure_sync_model = "container_pts_alignment_not_physical_exposure_genlock"`, recording that container timestamp alignment does not calibrate CMOS rolling-shutter readout or exposure integration windows.

Furthermore, **complete synchronized pair validity** (`valid_pair`) requires both frames to exist **and** the reference timestamp to fall inside the common overlap interval (`valid_pair = bool(valid_back and valid_front and in_overlap)`), while **per-camera update eligibility** is decoupled (`should_update_back_state = bool(is_new_b and valid_back)`, `should_update_front_state = bool(is_new_f and valid_front)`) so either camera can independently process a valid new physical frame even when the opposite camera's frame is unavailable.

### Condition 4: Production-Path Demonstration That Repeated 30 fps Frames Do Not Trigger Duplicate Updates
In [`test_repeated_30fps_frame_does_not_trigger_duplicate_detection_or_temporal_update`](file:///Users/sayemuddin/Desktop/research_helmet/tests/test_phase12c_pts_sync.py#L140-L220), calling the production `SynchronizedVideoReader.get_synchronized_pair(t_ref)` on `pair_015` (60 fps Front + 30 fps Back) for 8 canonical 60 Hz ticks produces `8` unique Front updates (`should_update_front_state = True` on all 8 ticks) and **exactly `4` unique Back updates** (`is_new_back_frame = True` and `should_update_back_state = True` on 4 ticks, `False` on 4 repeated ticks).
- Passing production `status["should_update_back_state"]` into [`DeduplicatedTemporalGate`](file:///Users/sayemuddin/Desktop/research_helmet/src/synchronization/synchronized_reader.py#L72-L103) executes the Back detector and Phase 10 `TemporalStateMachine` (`confirm_frames = 5`) **exactly 4 times** (`consecutive_no_helmet_frames == 4`, remaining in `TemporalState.NO_HELMET_SUSPECTED`).
- Without the gate, feeding the repeated 30 fps frames on all 8 ticks falsely advances `consecutive_no_helmet_frames` to `8` and triggers a premature `TemporalState.NO_HELMET_CONFIRMED` violation.
- In [`test_decoupled_per_camera_update_eligibility_and_overlap_validity`](file:///Users/sayemuddin/Desktop/research_helmet/tests/test_phase12c_pts_sync.py#L223-L270), querying `pair_003` outside the common overlap (`t_back = 0.10 s`, where `frame_f is None`) verifies `is_in_overlap == False`, `valid_pair == False`, `should_update_front_state == False`, and `should_update_back_state == True` on new physical Back frames (`False` on repeated Back frames).

---

## 4. Test Suite & Apple M1 Hardware Resource Measurements

- **Complete Test Suite**: `35 passed` (`28` Phase 12A baseline tests + `7` Phase 12C PTS synchronization, overlap validity, and decoupled deduplication tests).
- **Full 50-Video / 25-Pair Forensic Audit Runtime**: `158.88 s` (~`3.18 s` per 4K video stream including complete sequential `grab()` verification).
- **Peak Resident Memory (RSS) on Apple M1**: **`477.06 MB`** (well below the `1,850 MB` ceiling).
