# SONNET TASK-12 — Blind Dual-Camera Human GT v1.1 Review Dashboard Report

**Project:** Dual-Camera Context-Aware Motorcycle Helmet Violation Detection  
**Branch:** `main`  
**Target Package:** `data/dual_camera_test_v1/releases/dual_camera_gt_v1.1_blind_human/`  
**Human Verification Status of Checked-in Package:** `PENDING_HUMAN_REVIEW` (`human_verified: false`, `executed_human_units: 0`, `fabricated_labels_present: false` — strictly preserved)

---

## 1. Architecture & Chosen Stack

To guarantee fast, deterministic local execution on an 8 GB M1 MacBook Air without introducing unverified third-party web framework dependencies:

- **Backend (`src/review_dashboard/`)**:
  - `server.py`: Python 3.11 standard-library `ThreadingHTTPServer` (`ReviewDashboardHTTPServer`) serving static UI assets, strict role-isolated REST APIs, HTTP `206 Partial Content` byte-range MP4 streams, and single-frame PTS inspection endpoints.
  - `storage.py`: Transactional SQLite (`PRAGMA journal_mode=WAL`) persistence with separate `reviewer_annotations`, `submission_history` (with SHA-256 checksums), `adjudications`, and append-only `audit_log` tables, plus atomic temp-file-and-replace (`os.replace`) JSON sidecar snapshots.
  - `blind_api.py`: Read-only catalog loader over `blind_annotation_workqueue.csv`, `pair_manifest.csv`, and `synchronization_metadata.json`, enforcing strict allowlist sanitization (`REVIEWER_ALLOWED_ITEM_KEYS`), HTTP `403` peer/adjudicator isolation, field-level disagreement diffing, and dependency-free Cohen's $\kappa$ calculation.
  - `video_service.py`: Memory-bounded streaming & single-frame inspection (`BoundedVideoService`) leveraging existing `src/synchronization/pts_index.py` (`FramePTSIndex`), bounded LRU PTS cache (`max_pts_index_pairs=2`), single active OpenCV capture pair (`open_captures <= 2`) with deterministic `.release()`, and optional on-demand 720p H.264 browser fallback proxies (`max_proxy_pairs=2` LRU cache).
  - `validation.py` & `models.py`: Schema validation, cross-entity referential integrity (`rider -> motorcycle`, `event -> motorcycle + matching riders`), timeline interval validation (supporting both negative reference offsets and 1-frame point events `[t, t]`), and deterministic `v1.1` `schema.json` summary derivation.
- **Frontend (`src/review_dashboard/static/`)**:
  - Vanilla HTML5 (`index.html`), CSS (`styles.css`), and ES6 JavaScript (`app.js`) — zero bundler/Node build required at runtime.

---

## 2. Local Startup Command & Access URL

```bash
./scripts/start_gt_review_dashboard.sh
```

Default URL: **http://127.0.0.1:8765/**

Optional CLI flags / environment overrides:
- `--host 127.0.0.1` (`GT_REVIEW_HOST`)
- `--port 8765` (`GT_REVIEW_PORT`)
- `--workspace-dir <path>` (`GT_REVIEW_WORKSPACE_DIR`)
- `--db-path <path>` (`GT_REVIEW_DB_PATH`)
- `--no-proxy` (disable on-demand H.264 proxy transcoding)
- `--verbose-http` (enable per-request HTTP logs)

Stop cleanly at any time with **`Ctrl+C`**; `server_close()` deterministically releases active OpenCV video captures and closes the server socket.

---

## 3. Enforced Blindness & Protocol Alignment

1. **Zero AI / Prior Ground Truth Exposure**: `src/review_dashboard` never imports inference models or reads `dual_camera_gt_v1.0_ai_assisted_audited`, Stage J/K outputs, YOLO/ResNet/OCR predictions, association scores, or tracking IDs.
2. **Workqueue Leakage Stripping (`REVIEWER_ALLOWED_ITEM_KEYS`)**: Reviewer APIs strip `review_focus` (e.g., `pillion_presence_and_helmet_blind_check`, `resolve_insufficient_evidence`), `source_origin` (`GAPC_UNLINKED_OR_SPLIT_CANDIDATE`), and hintful `priority_stratum` names (`STRATUM_1_GAPC_PILLION_OR_CONFLICT` -> neutral `STRATUM_1`), exposing only workqueue/pair identifiers, neutral sampling bucket codes, sync timing, and temporal search windows (`front_transit_ref_sec_hint`, `back_transit_ref_sec_hint`) needed to locate each transit on multi-transit clips.
3. **Strict Reviewer Isolation**:
   - Requests use `X-Reviewer-Id` header or the local session cookie (`gt_review_actor`). The UI explicitly marks this as a local workstation profile selector (not cryptographic multi-user authentication).
   - `reviewer_a` and `reviewer_b` cannot query each other's drafts, submissions, snapshots, history, or exports (peer query parameter spoofing or cross-snapshot fetches return `HTTP 403`).
   - All `/api/adjudication/*` routes return `HTTP 403` to any blind reviewer.
4. **Schema Enum Alignment (`v1.1` Protocol Preservation)**:
   - UI helmet choices `HELMET`, `NO_HELMET`, `UNCERTAIN`, `NOT_VISIBLE` preserve fine-grained human visibility states in stored payloads while deterministically mapping `HELMET -> helmet`, `NO_HELMET -> no_helmet`, and `{UNCERTAIN, NOT_VISIBLE} -> occluded_unknown` to conform with `dual_camera_gt_v1.1_blind_human/schema.json`.
   - Temporal bounds allow negative `t_ref` timestamps when `global_offset_sec < 0` (`t_ref >= min(0.0, global_offset_sec)`, e.g., `pair_001` starts front at `-0.205s` and `pair_014` at `-1.820s`).

---

## 4. Video Codec & Synchronized Playback Behavior

All 50 source videos (25 front, 25 back, `3840x2160` 4K ~60 fps) use:
- **Front camera (`25/25`)**: `H.264` (`avc1`, `High`, `yuv420p`)
- **Back camera (`25/25`)**: `HEVC` (`hvc1`, `Main`, `yuv420p`)

Observed browser playback & synchronization behavior:
- **Sync equation**: Back camera normalized PTS serves as reference timeline `t_ref`. Front camera time is computed continuously as `t_front = t_ref - global_offset_sec`. Streams outside `[0, duration]` show an explicit range overlay (`FRONT/BACK STARTS AT...` or `ENDED AT...`) while the active camera continues smoothly.
- **Frame stepping**: `[` / `]` (or `Left` / `Right`) query `/api/frame_info/<pair_id>?step_frames=±1` to advance by exact container packet PTS timestamps using `FramePTSIndex`.
- **Automatic Hardware / Software Fallback**: Apple Silicon macOS hardware acceleration (Chrome / Safari GUI) decodes both 4K `avc1` and `hvc1` natively at `3840x2160`. When a browser session lacks HEVC video decoding (e.g., GPU disabled where `<video>` loads container metadata with `videoWidth === 0` or fires a decode error), `app.js` automatically switches to `?mode=proxy_h264`, lazily transcoding bounded `1280x720` H.264 previews inside `review_workspace/.proxy_cache/` (LRU cap = 2 pairs) without touching original `.mp4` files.

---

## 5. User Guides

### A. Blind Reviewer Guide (`Reviewer A` / `Reviewer B`)
1. Open `http://127.0.0.1:8765/` and choose **Reviewer A** or **Reviewer B** in the top Session Profile selector.
2. Click **Resume Work** (Overview) or open **Annotation Queue** to filter the 92 items across 25 pairs.
3. In **Video Review**:
   - Play/pause synchronized views (`Space`), step frames (`[` / `]`), scrub the unified timeline, or toggle **Loop Hint (`L`)**.
   - Zoom (`+` / `-`) and drag to pan inside either viewport; click **4K Inspect** for a full-resolution PTS-exact PNG extract or **Snap** to save a keyframe image into your isolated snapshot folder.
   - Add Motorcycles (`M1`...), Rear Plate Visibility & Text (`PLATE_CLEAR` requires normalized plate string), Riders linked to motorcycles (`DRIVER`/`PILLION`/`UNCERTAIN`), and Temporal Violation Events (supporting `[t_start, t_end]` windows or 1-frame `[t, t]` point events).
   - For non-motorcycle targets (e.g., pedal bicycle / pedestrian), check **Not a Motorcycle Transit** and explain in **Reviewer Notes**.
4. Drafts autosave to browser `localStorage` and SQLite after 1.2s of inactivity (`Ctrl/Cmd+S` forces immediate save).
5. Click **Validate & Submit Final** when ready. Submitted items are permanently locked and backed up to `submissions/<reviewer>/<workqueue_id>.json`.

### B. Adjudication Guide (`Adjudicator` Role)
1. Only switch to **Adjudication Coordinator** after independent reviews are complete.
2. Open **Adjudication**: items remain locked (`HTTP 409`) until **both** `Reviewer A` and `Reviewer B` have submitted immutable records.
3. For any both-submitted item, click **Adjudicate** to inspect side-by-side submissions, highlighted disagreements, and dual-camera videos. Save `ADJUDICATED` (with final schema fields) or `UNRESOLVED` (escalated cases).
4. Review real-time Cohen's $\kappa$ metrics (`binary_violation`, `rider_helmet`, `cross_camera_link`) across paired submissions.

### C. Coordinator Storage, Backup & Export Guide
- **Runtime Workspace**: Stored inside `data/dual_camera_test_v1/releases/dual_camera_gt_v1.1_blind_human/review_workspace/` (SQLite WAL DB + per-submission JSONs + snapshots + exports + backups; gitignored runtime artifacts).
- **Safe Backups**: Click **Backup DB** in the top bar (or `POST /api/system/backup`) to generate a hot consistent SQLite `.backup()` snapshot under `review_workspace/backups/`.
- **Exporting v1.1 Deliverables**: As Adjudicator, click **Export Complete v1.1 Bundle** (or `GET /api/adjudication/export`) to write schema-validated `release_metadata_snapshot.json`, `blind_annotation_workqueue_populated.csv`, `annotations_reviewer_a.jsonl`, `annotations_reviewer_b.jsonl`, `adjudicated_annotations.jsonl`, and `agreement_report.json` under `review_workspace/exports/`. Root checked-in `release_metadata.json` (`PENDING_HUMAN_REVIEW`) and `blind_annotation_workqueue.csv` are never altered automatically.

---

## 6. Verification & Integrity Summary

Verified against the current code state after Audit Pass 1 & Pass 2 remediations before commit:
- **Full Repository Pytest Suite**: `/opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest -q` -> **232 passed** (0 failures, 0 skipped; includes all 10 `tests/test_phase12_review_dashboard.py` tests plus all existing repository tests).
- **Dashboard & API Test Suite**: `/opt/homebrew/opt/python@3.11/bin/python3.11 -m pytest -v tests/test_phase12_review_dashboard.py` -> **10 passed** (covering reviewer separation, zero AI/peer leakage, negative-offset & `[t, t]` point-event validation, strict explicit-choice submission & non-finite number rejection, multi-entity `derive_item_summary` alignment, temporal violation-window disagreement detection, 3-class `binary_violation` + all-rider `rider_helmet` Cohen's $\kappa$, reload persistence, submission locks & duplicate rejection, `HTTP 409` adjudication gating, adjudication `[min_t_ref_sec, max_t_ref_sec]` & contradiction guards, `UNRESOLVED` & `COMPLETE_ADJUDICATED` export transitions, HTTP `206` Range video streaming, bounded OpenCV capture cleanup, negative-offset cross-camera boundary frame stepping, unique rapid snapshots, malformed route/query `4xx` handling, and `720p` H.264 proxy fallback).
- **Frontend JS Syntax Check**: `node --check src/review_dashboard/static/app.js` -> exited `0`.
- **Audit Pass 2 Exhaustive Backend Verification (92 Items / 25 Pairs)**:
  - Verified all `92/92` workqueue items (`0` AI/peer leakage keys in reviewer APIs), all `25/25` video pairs, all `13/13` negative-offset cross-camera boundary `-1f`/`+1f` frame steps, end-to-end submission + `18` disagreed/adjudicated items + export bundle (`executed_human_units: 92`, `human_verification_status: COMPLETE_ADJUDICATED`, `human_verified: true`), and `schema.json` validation.
- **Audit Pass 2 Headless Chrome CDP End-to-End UI Verification**:
  - Verified live browser interactions against `src/review_dashboard/static/` (`0` JS runtime exceptions, `0` HTTP `5xx` responses): dual-camera hint display on `WQ_010`, explicit empty-select submit blocking, motorcycle link synchronization, authenticated `4K Inspect` PNG modal load & frame telemetry header decode, negative-offset `-1f`/`-3f` stepping + Back `Snap` out-of-range guard on `WQ_001`, `loadedmetadata` initial seek alignment on `WQ_015`, conflicting Reviewer A/B submissions on `WQ_003`, and Adjudicator `.field-disagree` highlighting with blanked disagreed fields.
  - Prior baseline (`e1d817c`) Chrome CDP checks additionally verified 4K simultaneous native hardware decode (`3840x2160` `avc1` + `hvc1`, `readyState: 4`) and `--disable-gpu` auto-fallback (`1280x720` H.264 proxy playback, `proxyChecked: true`).
- **Exhaustive 50-Video / 92-Workqueue-Item Technical Readiness Verification** (`video_service.py` / `pts_index.py` unchanged since initial verification):
  - Initial readiness verified all `50/50` MP4 files (`25` front, `25` back) against `pair_manifest.csv` SHA-256 checksums (`0` mismatches), decoded all `15,241/15,241` physical frames sequentially at `3840x2160` with monotonic container PTS indices, achieved `100/100` mid/final frame checks at `0.0000` mean pixel diff via cursor-tracked sequential `cap.grab()`, and verified `154` hint frames across `92/92` workqueue items.
- **Frozen Research Artifact & Regression Integrity**:
  - `scripts/reproducibility/event_regression_task04.py`: `events compared=85 changed=0`.
  - Frozen `v1.0` release checksums (`data/dual_camera_test_v1/releases/dual_camera_gt_v1.0_ai_assisted_audited/checksums/release_checksums.sha256`): **43/43 OK**.
  - Base annotation checksums (`data/dual_camera_test_v1/annotations/checksums/annotation_checksums.sha256`): **39/39 OK**.
  - Phase 12 integrated checksums (`results/phase12_integrated/checksums.sha256`): **8/8 OK**.
  - Checked-in `v1.1` package unchanged (`PENDING_HUMAN_REVIEW`, `human_verified: false`, `executed_human_units: 0`, `fabricated_labels_present: false`, `blind_annotation_workqueue.csv` MD5 `82d9e890f64711b5c303a06e711ddf30`), and `review_workspace/` contains only `README.md` (zero synthetic or test artifacts).

---

## 7. Audit Pass 1 & Pass 2 Remediation Summary

Two full re-audit passes across backend validation/API/storage/video-stepping/HTTP server (`src/review_dashboard/`) and frontend UI (`src/review_dashboard/static/`) identified and resolved the following pre-review bugs:

1. **Strict Validation & Non-Finite Input Rejection (`validation.py`, `server.py`)**:
   - Removed silent default fallbacks on `submit` (`cross_camera_link_status`, `role`, `helmet_state`, `event_type`, `camera_evidence`) so reviewers must make explicit selections, while preserving blank values in `draft`.
   - Required `visible_front or visible_back` on `submit`, enforced cross-field consistency between link status, motorcycle visibility, rider evidence, event evidence, and motorcycle consistency, and corrected multi-pillion aggregation (`_aggregate_helmet_labels`) plus multi-motorcycle `derive_item_summary` alignment.
   - Replaced `_as_float` coercion with `_parse_optional_float` (`math.isfinite`) and added strict JSON `parse_constant` / `allow_nan=False` guards in `server.py` so `NaN` / `Infinity` payloads or query parameters return `HTTP 400`/`422`.
2. **Adjudication Disagreement, IAA Kappa & Export Accounting (`blind_api.py`, `storage.py`)**:
   - Extended `compute_reviewer_disagreements` to detect temporal `"violation_window_ref_sec"` conflicts (`_parse_window_ref`, `_violation_windows_disagree`, `point_tol_sec=0.15`, `min_tiou=0.5`).
   - Updated `get_adjudication_overview` so `binary_violation` Cohen's $\kappa$ separates 3 classes (`VIOLATION`, `NON_VIOLATION`, `INSUFFICIENT_EVIDENCE`) and `rider_helmet` Cohen's $\kappa$ evaluates both `driver_helmet` and `pillion_helmet`.
   - Enforced `[min_t_ref_sec, max_t_ref_sec]` and cross-field guards on `ADJUDICATED` saves (`save_adjudication`), restricted reviewer/adjudicator JSONL exports to `SUBMITTED`/`ADJUDICATED` records, excluded `UNRESOLVED` items from `executed_human_units` and completion checks, and corrected `human_verification_status` transitions (`PENDING_HUMAN_REVIEW` vs `IN_PROGRESS` vs `COMPLETE_ADJUDICATED`).
   - Added microsecond + random hex uniqueness to `save_reviewer_snapshot_image` filenames (`%Y%m%dT%H%M%S_%f_{hex}Z`) to prevent same-second overwrites.
3. **Cross-Camera Frame Stepping, HTTP Keep-Alive & Server Hardening (`video_service.py`, `server.py`)**:
   - Updated `step_reference_time` with cross-camera boundary handoff so stepping `-1f` / `+1f` across `t_ref = 0.0` on negative-offset pairs transitions cleanly between Back Frame `0` and the last pre-zero Front frame (exact round-trip inverse).
   - Validated `fmt`/`max_width`/`crop` in `extract_frame_data`, tightened 2-segment `/api/reviewer/items/` routing, added `self.close_connection = True` + `"Connection": "close"` in `_send_error_json` (eliminating unread-body keep-alive framing errors), and suppressed benign client seek/abort `BrokenPipeError`/`ConnectionResetError` tracebacks in `ReviewDashboardHTTPServer.handle_error`.
4. **Frontend UI Alignment & Media Sync (`index.html`, `styles.css`, `app.js`)**:
   - Added separate `#vis-front-hint-band` / `#vis-back-hint-band` overlays with union hint seeking (`getHintInterval`), explicit `-- Choose --` options on reviewer/adjudicator selects, automatic `syncMotorcycleConsistency`, `loadedmetadata` initial seek replay (`syncVideosToRefTime`), effective `<video>` duration bounds for out-of-range Snapshot blocking, autosave timer cancellation on item switch/submit, authenticated `fetch` + headers in `inspectFrameModal`, `.field-disagree` visual highlighting + blanking of disagreed adjudicator fields, and proxy fallback/unload cleanup on `#adj-video-*`.

---

## 8. Known Limitations

1. **Local Workstation Session Selector**: Role isolation separates local SQLite records and API responses cleanly via cookie/header session state (`reviewer_a` / `reviewer_b` / `adjudicator`) for single-workstation lab use; it does not implement password login or multi-user remote TLS authentication.
2. **4K Frame-Exact Seeking Speed on Long Clips**: Browser playback uses macOS GPU hardware decoding (or bounded 720p H.264 proxy when fallback activates), whereas OpenCV single-frame extraction (`4K Inspect` / `Snap`) uses cursor-tracked sequential `cap.grab()` instead of `CAP_PROP_POS_FRAMES` to guarantee exact VFR H.264 / B-frame HEVC alignment; forward stepping and repeated inspect/crop/snapshot on the current frame are near-instant, while cold jumps or backward steps near the end of long 4K clips take up to ~3–6s on CPU.
3. **Human Review Pending**: Per research integrity rules, zero human labels have been pre-filled; `dual_camera_gt_v1.1_blind_human` awaits real annotation execution by Reviewer A, Reviewer B, and the Adjudicator.
