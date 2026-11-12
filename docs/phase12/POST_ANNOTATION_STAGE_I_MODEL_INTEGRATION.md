# Post-Annotation Stage I — Model-Integration Verification

**Date:** 2026-11-09 · **Stage:** I of the post-annotation mission (Stages A–H complete)
**Subject:** Stage G pipeline (`src/pipeline/dual_camera_pipeline.py`) × the REAL models
**Evidence:** `work/stageI/` (STAGE_I_PROGRESS.md, weight_sha256.txt, registry_check.json,
run_evidence.json, output_audit.json, run_pair_001/ artifacts, kaggle_*_list.txt)

## Verdict

**Model integration is VERIFIED.** Both weights are bit-identical to the recorded
hashes; both models' class mappings are exactly as the Phase 12D configuration
specifies, with load-time verification at three independent layers; the detector
registry demonstrably loads each model once and shares it across cameras; and a
real end-to-end run of the Stage G pipeline on pair_001 reproduced Pass 1's
pair_001 results **exactly** — detection totals identical in every category on
both cameras (delta 0), identical track and association outcomes, all 8 summary
validation checks passed. **No pipeline defect was found; `src/pipeline/` was not
modified** (so no Stage H re-run was triggered).

## 1. Weight integrity (S1)

| Weight | Bytes | SHA-256 (computed) | SHA-256 (recorded) | Verdict |
|---|---|---|---|---|
| `results/phase9_multiseed/exp_combined/seed_42/weights/best.pt` | 6,506,922 | `0929a882…73d840b0e5` | `0929a882…73d840b0e5` | MATCH |
| `yolov8n.pt` (repo root) | 6,549,796 | `f59b3d83…fc83b36` | `f59b3d83…fc83b36` | MATCH |

Full hashes in `work/stageI/weight_sha256.txt`. The same two hashes are also
pinned in `configs/tracking_config.yaml` (`detection.*.sha256`), so config,
registry records, and files on disk all agree.

## 2. Class mappings (S2)

Live `model.names` dumps (via the adapters, `work/stageI/registry_check.json`):

- **yolov8n.pt:** 80 COCO classes; `[0] = person`, `[3] = motorcycle` — exactly the
  entries the pipeline consumes.
- **best.pt (Phase 9):** `{0: no_helmet, 1: helmet}` — cranial classes only.

The mapping is enforced, not assumed, at three layers:

1. **Config** — `tracking_config.yaml` pins `target_classes {0: person, 3: motorcycle}`
   and `{0: no_helmet, 1: helmet}` alongside the weight hashes.
2. **Adapters** — `VehiclePersonDetectorAdapter.__init__` raises if the loaded
   model's names for IDs 0/3 differ from `person`/`motorcycle`, and its inference
   is class-filtered at the source (`classes=[0, 3]`), so no other COCO class can
   ever enter the pipeline. `HelmetDetectorAdapter` exposes the model's own names.
3. **PerCameraDetector** — its constructor hard-asserts the helmet adapter's
   `class_names == {0: "no_helmet", 1: "helmet"}` and raises otherwise (this
   assertion executed and passed in both Stage I processes).

Mapping trace in `PerCameraDetector.detect_frame` (src/detection/camera_detector.py):
vehicle detections are bucketed strictly by class id (`3 → motorcycles`,
`0 → persons`; the category string is derived from the same id, and appends are
gated on the id, so no other id can be bucketed — and per layer 2 none can
arrive). Helmet-model detections become `category="head"` with
`cid == 1 → helmet_boxes`, else `no_helmet_boxes`; given the layer-3 assertion,
`else` is provably class 0 = `no_helmet`. **No off-by-one or swapped-label path
exists.** Post-hoc proof from the run itself: the S5 audit checked every one of
the 2,054 detection records in the pair_001 output JSONL for
(class_id, class_name, category, source_model) consistency — **0 violations**
(§4).

## 3. Registry singleton behaviour (S3)

Instrumented, not assumed (wrap-only counters; no source modified):

- Building `PerCameraDetector("front")` and `PerCameraDetector("back")`
  (device `cpu`) in a fresh process constructed YOLO **exactly once per model**
  (vehicle 1, helmet 1). The two cameras' detectors hold the **identical**
  adapter objects (`is` equality) and the **identical** underlying `.model`
  objects; the registry caches hold exactly one entry each (key `cpu`).
- The instrumented real run (§4) confirms the same inside the pipeline itself:
  adapter constructions during `run_pair` = **1 vehicle + 1 helmet** for the
  whole pair, while `detect_frame` was called 206 (front) + 397 (back) times.

## 4. Real end-to-end run — pair_001 (S4/S5)

Config: `PipelineConfig` with the real dataset root, pinned Phase 11 metadata,
real tracking/association configs, `device="cpu"`, output
`work/stageI/run_pair_001/`. Pair_001 front is forensically VFR (59.72 fps
effective), back 59.92 fps — the run exercises the VFR/new-frame path with real
models.

**Outcome: COMPLETE, wall 160.7 s for 603 frames (0.267 s/frame, both models),
peak RSS ≈ 981 MB. Summary validation: 8/8 checks passed** (coverage complete
both cameras, detections PTS match table, association counts reconcile,
association references known tracklets, events cover all moto tracks, sync
diagnostics all Q-fields present).

### 4.1 New-frame-only with real models

| Camera | Decoded frames | Coverage rows | Detection JSONL lines | `detect_frame` calls |
|---|---|---|---|---|
| front | 206 | 206 | 206 | **206** |
| back | 397 | 397 | 397 | **397** |

Exactly one detector invocation per unique decoded frame per camera — no
duplicate presentation was processed (primary loop; `duplicate_ticks_skipped`
= 0 on both cameras, `frames_updated` = 206/397 in the track artifacts).

### 4.2 Detection totals vs Pass 1 (same models, same pair)

Pass 1 ran on a Kaggle T4; this run on local CPU. Totals counted identically
from each run's detection JSONL:

| Camera | Category | Stage I | Pass 1 | Δ |
|---|---|---|---|---|
| front | motorcycle | 191 | 191 | **0** |
| front | person | 538 | 538 | **0** |
| front | helmet | 6 | 6 | **0** |
| front | no_helmet | 283 | 283 | **0** |
| back | motorcycle | 166 | 166 | **0** |
| back | person | 669 | 669 | **0** |
| back | helmet | 0 | 0 | **0** |
| back | no_helmet | 201 | 201 | **0** |

(Back-camera helmet = 0 with 201 no_helmet heads is the known rear-view profile
of this pair, identical in both runs — detector behaviour, not an integration
fault.) Tracks and association reproduce Pass 1 as well: front 1 moto / 7
orphan tracks, back 1 moto / 3 orphan tracks; moto association UNMATCHED ×2;
orphan outcomes UNMATCHED 3 / AMBIGUOUS_ABSTAIN 3 / INSUFFICIENT_EVIDENCE 3;
rider outcomes all zero. The 2 derived events are well-formed (Phase 10 alert
IDs attached, rider states with `AI_DIRECT` provenance, plates `NOT_ASSESSED`).

### 4.3 Sync diagnostics in the run

`pair_001_sync_diagnostics.json` carries the full Q-field set: `offset_sec`
−0.205, `epsilon_sync_ms` 3.0, **`drift_ms` 0.0** (reported, not applied —
GAP-1), and **true Q5** overlap lookup statistics (`q5_overlap_lookup_stats`:
187 samples; front mean 5.44 ms / max 11.54 ms; back mean 1.80 ms / max
3.33 ms) kept strictly separate from `nominal_grid_deviation_ms` (GAP-4), with
an in-artifact `delta_pts_field_definitions` block and the exposure-sync
disclaimer (`container_pts_alignment_not_physical_exposure_genlock`).

## 5. Stage J readiness plan

**Venue decision: Kaggle T4 (account `elafahmed6563`), following the Pass 1
pattern** — as pre-registered in the Stage F design (OQ-4). Reasoning, measured:

- Pass 1 processed the full 15,241-frame corpus in ~11 min of GPU camera time
  (644 s) with these same models; a Stage J kernel should complete in roughly
  15–25 min of T4 time including per-pair overheads.
- Local CPU is a feasible but slower fallback: this run measured 0.267 s/frame
  → ≈ 68 min of pure inference for the corpus at today's contention, and
  Stage G's contended benchmark (671 ms/frame) implies ≈ 2.8 h. The VM's
  ~1.5-core quota makes the local figure unreliable.

**Staging (what carries over, verified 2026-11-09):**

- Dataset `elafahmed6563/helmet-annotation-inputs` is **still hosted**
  (2,182,327,905 B; Kaggle CLI auth verified live from the VM in this stage) —
  the 50 videos + Phase 11 metadata need no re-upload.
- A **refreshed `repo_code.zip`** must be built from the current repo (the
  Pass 1 zip predates `src/pipeline/`): include `src/` (pipeline + all approved
  modules), `configs/`, and both weights at their pinned paths (weights are
  already in the current zip at the correct paths/sizes). Ship it as a new
  version of the same dataset (Pass 1 pattern) or a small companion dataset.
- New kernel (script type, T4, internet on, private) modelled on
  `helmet-pass1-annotation`: use the extracted-tree layout (Pass 1 lesson:
  Kaggle extracts zips server-side — never assert the zip file exists), run the
  pipeline per pair via its CLI `--worker` mode or `run_all`, with
  `device="cuda"`, and emit the per-pair output tree + `run_manifest.json` as
  kernel output. Outputs must be downloaded and hash-verified on the VM before
  any pair is declared complete (Pass 1 pattern).

**Per-pair acceptance gates for Stage J** (all demonstrated in this stage):
summary `COMPLETE` with `validation.all_passed == true` (8/8 checks); decoded
frames == Phase 12C forensic counts per camera (206/397 pattern, pair-specific);
detection JSONL lines == decoded frames; batch manifest reconciles 25/25 pairs
with failure isolation (a failed pair is re-run, never silently skipped);
resume only via the hash-validated checkpoint rule.

**Quota:** the Kaggle CLI exposes no remaining-GPU-quota figure (checked in
this stage — list operations succeed, no quota endpoint). **Stage J pre-flight
must confirm the account's GPU quota in the Kaggle web UI before pushing the
kernel**; expected consumption is ≈ 0.3–0.5 h of T4 time, small against the
standard 30 h/week quota, but the account has run kernels for other projects
today, so the check is a real gate, not a formality. Do not push any kernel
until it passes.

## 6. Notes and limitations

- Count-level equality with Pass 1 (§4.2) is exact on this pair; box-level
  identity was not asserted (CPU vs T4 float paths could in principle flip a
  borderline box on another pair — Stage K evaluates against the frozen GT
  regardless, and Stage J's gates are count/coverage-level by design).
- The VM venv prints a pre-existing torchvision/torch version warning on
  import; it does not affect inference (both models loaded and ran) and is
  unchanged from earlier stages.
- No changes were made in this stage to weights, configs, approved modules,
  the pipeline, the frozen package/release, or tests.
