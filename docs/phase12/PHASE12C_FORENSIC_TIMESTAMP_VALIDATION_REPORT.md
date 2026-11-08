# PHASE 12C — FORENSIC TIMESTAMP & PTS ALIGNMENT VALIDATION REPORT

**Project:** An Integrated Dual-Camera Framework for Context-Aware Motorcycle Helmet Violation Detection with Confidence-Aware Temporal Verification  
**Dataset Audited:** `data/dual_camera_test_v1/` (50 Video Containers across 25 Synchronized Pairs, `pair_001`–`pair_025`)  
**Generated Artifacts:**  
- [`results/phase12c_pts_sync/stream_pts_forensics_50_videos.csv`](file:///Users/sayemuddin/Desktop/research_helmet/results/phase12c_pts_sync/stream_pts_forensics_50_videos.csv)  
- [`results/phase12c_pts_sync/pair_pts_alignment_validation_25_pairs.csv`](file:///Users/sayemuddin/Desktop/research_helmet/results/phase12c_pts_sync/pair_pts_alignment_validation_25_pairs.csv)  
- [`results/phase12c_pts_sync/phase12c_pts_validation_summary.json`](file:///Users/sayemuddin/Desktop/research_helmet/results/phase12c_pts_sync/phase12c_pts_validation_summary.json)

---

## 1. Stream-Level Container Forensics (50 Videos)

### 1.1 Validation of `CAP_PROP_POS_MSEC` vs. `ffprobe` Rational PTS vs. `CAP_PROP_POS_FRAMES`
Across all 50 video containers (`25 front + 25 back`, totaling `15,637` frames):
1. **Frame Count Integrity**: `50 / 50` streams have an exact `1:1` match between `ffprobe` video packet count and OpenCV sequential decoded frame count (`all_50_frame_counts_exact_match = True`).
2. **Sequential `CAP_PROP_POS_MSEC` vs. Stream-Normalized Rational PTS**:
   - When frames are decoded sequentially from the start of the stream via `cap.grab()`, OpenCV's `CAP_PROP_POS_MSEC / 1000.0` agrees with stream-normalized rational PTS $t_{\text{norm}}[k] = (\text{PTS}[k] - \text{PTS}[0]) \cdot \frac{b_{\text{num}}}{b_{\text{den}}}$ to **`0.000000 ms`** across all 50 files.
   - However, on 24 of 25 Android Front streams (`pair_002_front` through `pair_025_front`, timebase `1/90000`), unnormalized `ffprobe` `start_time` begins between `+31.3 ms` (`pair_024_front`) and `+79.8 ms` (`pair_023_front`) because the audio track starts slightly ahead of the first video packet, whereas `CAP_PROP_POS_MSEC` normalizes Frame 0 to `0.000000 s`.
3. **Failure of OpenCV Random Seeking (`cv2.CAP_PROP_POS_FRAMES`) on VFR Streams**:
   - Calling `cap.set(cv2.CAP_PROP_POS_FRAMES, k)` does **not** count decoded frames in OpenCV's FFmpeg backend; it converts `k` to a seek timestamp using average frame rate.
   - At the midpoint of `pair_013_back.mp4` (`frame_idx = 304`, true normalized PTS `5.105000 s`), `cap.set(cv2.CAP_PROP_POS_FRAMES, 304)` lands at `7.975000 s`—an actual decoded-frame timestamp error of **`2,870.00 ms` (2.87 seconds)**!
   - At the midpoint of `pair_014_back.mp4` (`frame_idx = 138`, true normalized PTS `2.300000 s`), `cap.set(cv2.CAP_PROP_POS_FRAMES, 138)` lands at `2.650000 s`—an error of **`350.00 ms`**.
   - At the midpoint of `pair_001_front.mp4` (`frame_idx = 103`, which has 1 dropped frame at index 29), `cap.set(cv2.CAP_PROP_POS_FRAMES, 103)` lands **`16.63 ms`** (1 frame) off.

### 1.2 Summary of VFR and Mixed-Frame-Rate Streams

| Stream Filename | Frame Count | Timebase | Effective FPS | VFR? | 60fps Intervals ($\sim 16.7\text{ ms}$) | 30fps Intervals ($\sim 33.3\text{ ms}$) | Nominal `idx/fps` Mean Error (ms) | Nominal `idx/fps` Max Error (ms) | `CAP_PROP_POS_FRAMES` Midpoint Seek Error (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `pair_001_front.mp4` | 206 | `1/90000` | `59.72` | **Yes** (1 drop) | 204 | 1 | `6.32` | `14.30` | `16.63` |
| `pair_013_back.mp4` | 608 | `1/19200` | `40.15` | **Yes (Severe)** | 308 | 299 | **`1,264.11`** | **`2,531.25`** | **`2,870.00`** |
| `pair_014_back.mp4` | 276 | `1/19200` | `48.47` | **Yes (Severe)** | 210 | 65 | **`412.67`** | **`829.03`** | **`350.00`** |
| `pair_015_back.mp4` | 190 | `1/19200` | `29.97` | No (30fps CFR) | 0 | 189 | `0.48` | `1.34` | `0.00` |
| `pair_016_back.mp4` | 310 | `1/19200` | `29.97` | No (30fps CFR) | 0 | 309 | `0.43` | `1.06` | `0.00` |
| `pair_017_back.mp4` | 200 | `1/19200` | `29.96` | No (30fps CFR) | 0 | 199 | `0.50` | `1.42` | `0.00` |
| `pair_018_back.mp4` | 198 | `1/19200` | `29.96` | No (30fps CFR) | 0 | 197 | `0.53` | `1.50` | `0.00` |
| *Remaining 43 CFR 60fps Streams* | 110–736 | `1/90000` or `1/19200` | `59.92–60.02` | No (60fps CFR) | All | 0 | `0.02–0.54` | `0.11–1.59` | `0.00` |

---

## 2. Pair-Level Synchronized Alignment & Separated Residuals (All 25 Pairs)

We maintain an explicit **four-way metrological distinction** across all validation artifacts and runtime status records:
1. **Container PTS Agreement**: Agreement between OpenCV sequential `CAP_PROP_POS_MSEC` and `ffprobe` stream-normalized rational PTS $t_{\text{norm}}[k]$ (`0.000000 ms` across all 50 streams).
2. **Frame-Selection Discretization Residuals (`delta_pts_front_ms` $\delta_{\text{pts},F}$, `delta_pts_back_ms` $\delta_{\text{pts},B}$)**: Discrete sampling offset $|t_{c,\text{actual}} - t_{c,\text{target}}|$ when stepping at a canonical `60.0 Hz` reference tick ($\Delta t_{\text{tick}} = 16.667\text{ ms}$). For example, in `pair_001` (per [`phase12c_pts_validation_summary.json`](file:///Users/sayemuddin/Desktop/research_helmet/results/phase12c_pts_sync/phase12c_pts_validation_summary.json)), `delta_pts_front_mean_ms = 5.4362 ms` (`max = 11.5444 ms` due to 1 dropped frame interval of `33.17 ms` at frame 29 in `pair_001_front.mp4`) while `delta_pts_back_mean_ms = 1.8004 ms` (`max = 3.3333 ms`).
3. **Phase 11.6 Inter-Camera Synchronization Residual (`epsilon_sync_ms` $\epsilon_{\text{sync}}$)**: Empirical inter-camera clock-alignment residual uncertainty (`1.0–10.0 ms` for the 21 sub-frame pairs; `18.0–38.0 ms` for the 4 near-frame pairs).
4. **Physical Exposure Synchronization**: Uncalibrated CMOS rolling-shutter readout and auto-exposure integration windows (`exposure_sync_model = "container_pts_alignment_not_physical_exposure_genlock"`), which container timestamp alignment does not equate to.

> [!NOTE]
> **Why `delta_pts_back_ms` and `delta_pts_front_ms` are NOT zero:**  
> Even with exact container PTS lookup, a camera records discrete frames every $16.67\text{ ms}$ (at 60 fps) or $33.33\text{ ms}$ (at 30 fps). When a reference tick $t_{\text{ref}}$ or offset-shifted target time $t_{\text{front}} = t_{\text{ref}} - \Delta t_{\text{rec}}$ falls between two discrete frame timestamps, the nearest frame is displaced by up to half a frame interval ($\le 8.33\text{ ms}$ at 60 fps; $\le 16.67\text{ ms}$ at 30 fps). Reporting `0.0 ms` for discrete frame matching across asynchronous clocks or mixed 30/60 fps streams would be scientifically false.

### Complete 25-Pair Validation Table (Canonical 60 Hz Reference Timeline)

| Pair ID | Sync Tier | $\epsilon_{\text{sync}}$ (ms) | FPS ($F$ / $B$) | 60Hz Ticks | Unique Frames ($F$ / $B$) | Suppressed Dup Ticks ($B$) | $\delta_{\text{pts},F}$ Mean / Max (ms) | $\delta_{\text{pts},B}$ Mean / Max (ms) | Inter-Cam PTS Skew Mean / Max (ms) | Legacy Nominal Back Max Error in Overlap (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `pair_001` | Sub-Frame | `3.0` | `59.7 / 59.9` | 187 | `186 / 187` | 0 | `5.44 / 11.54` | `1.80 / 3.33` | `7.24 / 11.54` | `3.33` (Front: `11.69`) |
| `pair_002` | Sub-Frame | `3.0` | `60.0 / 59.9` | 249 | `249 / 249` | 0 | `5.78 / 6.40` | `2.40 / 5.00` | `8.19 / 11.40` | `5.00` |
| `pair_003` | Sub-Frame | `3.0` | `60.0 / 59.9` | 233 | `233 / 233` | 0 | `0.46 / 0.96` | `4.75 / 6.67` | `5.20 / 7.62` | `6.67` |
| `pair_004` | Near-Frame | `18.0` | `60.0 / 59.9` | 101 | `101 / 101` | 0 | `2.19 / 2.42` | `0.94 / 1.67` | `3.13 / 4.09` | `1.67` |
| `pair_005` | Near-Frame | `35.0` | `60.0 / 59.9` | 205 | `205 / 205` | 0 | `0.44 / 0.88` | `4.07 / 6.67` | `3.62 / 6.69` | `6.67` |
| `pair_006` | Sub-Frame | `6.0` | `60.0 / 59.9` | 132 | `132 / 132` | 0 | `4.73 / 5.04` | `1.15 / 3.33` | `3.58 / 5.04` | `3.33` |
| `pair_007` | Sub-Frame | `4.0` | `60.0 / 59.9` | 156 | `156 / 156` | 0 | `0.31 / 0.61` | `3.69 / 5.00` | `3.99 / 5.61` | `5.00` |
| `pair_008` | Near-Frame | `38.0` | `60.0 / 59.9` | 155 | `155 / 155` | 0 | `0.31 / 0.72` | `2.58 / 3.33` | `2.27 / 3.50` | `3.33` |
| `pair_009` | Sub-Frame | `3.0` | `60.0 / 59.9` | 231 | `231 / 227` | 4 | `0.50 / 0.99` | `5.87 / 8.33` | `5.37 / 8.44` | `8.33` |
| `pair_010` | Sub-Frame | `7.0` | `60.0 / 59.9` | 109 | `109 / 109` | 0 | `0.35 / 0.61` | `4.40 / 5.00` | `4.75 / 5.61` | `5.00` |
| `pair_011` | Sub-Frame | `3.0` | `60.0 / 59.9` | 197 | `197 / 197` | 0 | `7.75 / 8.14` | `1.87 / 3.33` | `5.88 / 8.14` | `3.33` |
| `pair_012` | Sub-Frame | `3.0` | `60.0 / 59.9` | 237 | `237 / 236` | 1 | `0.55 / 1.07` | `7.16 / 8.33` | `6.66 / 8.59` | `8.33` |
| **`pair_013`** | **Sub-Frame (VFR)** | **`3.0`** | **`60.0 / 40.2`** | **403** | **`403 / 201`** | **202** | **`0.92 / 1.79`** | **`8.35 / 16.67`** | **`8.41 / 17.73`** | **`2,538.33`** *(152× worse)* |
| **`pair_014`** | **Sub-Frame (VFR)** | **`3.0`** | **`60.0 / 48.5`** | **156** | **`156 / 156`** | **0** | **`4.11 / 4.44`** | **`1.51 / 3.33`** | **`5.61 / 7.78`** | **`498.33`** *(149× worse)* |
| **`pair_015`** | **Near-Frame (30fps $B$)** | **`37.0`** | **`60.0 / 30.0`** | **250** | **`250 / 125`** | **125** | **`4.33 / 4.87`** | **`8.33 / 16.67`** | **`8.39 / 20.52`** | `16.67` (125 dup frames unflagged) |
| **`pair_016`** | **Sub-Frame (30fps $B$)** | **`10.0`** | **`60.0 / 30.0`** | **450** | **`450 / 225`** | **225** | **`6.96 / 7.99`** | **`8.33 / 16.67`** | **`11.23 / 24.66`** | `25.00` (225 dup frames unflagged) |
| **`pair_017`** | **Sub-Frame (30fps $B$)** | **`3.0`** | **`60.0 / 30.0`** | **199** | **`199 / 99`** | **100** | **`0.49 / 0.93`** | **`8.38 / 16.67`** | **`8.37 / 17.54`** | `16.67` (100 dup frames unflagged) |
| **`pair_018`** | **Sub-Frame (30fps $B$)** | **`3.0`** | **`60.0 / 30.0`** | **154** | **`154 / 77`** | **77** | **`0.49 / 0.88`** | **`8.36 / 16.67`** | **`8.37 / 17.46`** | `16.67` (77 dup frames unflagged) |
| `pair_019` | Sub-Frame | `8.0` | `60.0 / 59.9` | 229 | `229 / 229` | 0 | `2.24 / 2.82` | `2.18 / 5.00` | `1.45 / 3.26` | `5.00` |
| `pair_020` | Sub-Frame | `3.0` | `60.0 / 59.9` | 157 | `157 / 157` | 0 | `0.30 / 0.64` | `2.07 / 3.33` | `2.37 / 3.98` | `3.33` |
| `pair_021` | Sub-Frame | `3.0` | `60.0 / 59.9` | 272 | `272 / 272` | 0 | `0.88 / 1.47` | `2.62 / 5.00` | `3.50 / 6.47` | `5.00` |
| `pair_022` | Sub-Frame | `3.0` | `60.0 / 59.9` | 218 | `218 / 218` | 0 | `0.84 / 1.34` | `2.04 / 5.00` | `1.67 / 4.63` | `5.00` |
| `pair_023` | Sub-Frame | `3.0` | `60.0 / 59.9` | 318 | `318 / 318` | 0 | `2.64 / 3.30` | `3.04 / 6.67` | `1.98 / 4.73` | `6.67` |
| `pair_024` | Sub-Frame | `3.0` | `60.0 / 59.9` | 480 | `480 / 479` | 1 | `1.02 / 2.04` | `4.38 / 8.33` | `4.79 / 10.06` | `8.33` |
| `pair_025` | Sub-Frame | `3.0` | `60.0 / 59.9` | 208 | `208 / 208` | 0 | `7.24 / 7.69` | `1.95 / 3.33` | `9.19 / 11.02` | `3.33` |

---

## 3. Key Forensic Conclusions for Downstream Phases (12D–12G)

1. **VFR Catastrophe Eliminated**: On `pair_013`, the legacy nominal-FPS indexing error in the overlap window drops from a maximum of **`2,538.33 ms`** (and **`2,870.00 ms`** OpenCV `CAP_PROP_POS_FRAMES` seek error) to a bounded container discretization residual of **`mean = 8.35 ms, max = 16.67 ms`** (exactly half of a 30 fps frame interval). On `pair_014`, the overlap indexing error drops from **`498.33 ms`** to **`mean = 1.51 ms, max = 3.33 ms`**.
2. **Exact 30/60 FPS Deduplication**: Across the 5 pairs where the Back stream runs at 30 fps during the overlap (`pair_013`, `pair_015`, `pair_016`, `pair_017`, `pair_018`), `SynchronizedVideoReader` identifies all **`729` repeated 30 fps Back-camera ticks** (`is_new_back_frame == False`) and serves them from `VideoReader`'s in-memory frame cache with `should_update_back_state == False`.
3. **Preserved Uncertainty Separation**: Downstream cross-camera association (Phase 12E) and evidence fusion (Phase 12G) receive both the static pair-level synchronization residual (`epsilon_sync_ms`) and the dynamic per-frame PTS discretization offsets (`delta_pts_front_ms`, `delta_pts_back_ms`, `inter_camera_pts_skew_ms`).
