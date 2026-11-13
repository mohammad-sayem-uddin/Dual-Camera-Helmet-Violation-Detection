# Final Experimental Results & Statistical Synthesis

**Project:** An Integrated Dual-Camera Framework for Context-Aware Motorcycle Helmet Violation Detection with Confidence-Aware Temporal Verification  
**Accepted Active Dual-Camera Reference:** [`dual_camera_gt_v1.0_ai_assisted_audited`](file:///Users/sayemuddin/Desktop/research_helmet/data/dual_camera_test_v1/releases/dual_camera_gt_v1.0_ai_assisted_audited/README.md) (`AI_ASSISTED_AUDITED`, `human_verified = false`; `v1.1` preserved untouched as `PENDING_HUMAN_REVIEW`, `0 / 92` completed)

---

## 1. Phase 8 & Phase 9: Cross-Domain Detector Performance & Multi-Seed Reproducibility

### 1.1 Independent Single-Seed Reconciliation (Phase 8, Seed 42)
Verified from [`results/phase8_independent_audit/reconciled_metrics.csv`](file:///Users/sayemuddin/Desktop/research_helmet/results/phase8_independent_audit/reconciled_metrics.csv) and Stage 8 seed summaries:

| Model (Training Set) | Held-Out Test Set | Images | Instances | Precision | Recall | F1-Score | mAP@0.50 | mAP@0.50:0.95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **EXP-002-FIXED** *(CCTV only)* | Old CCTV | `829` | `1,170` | `0.7431` | `0.7228` | `0.7328` | `0.6694` | `0.2896` |
| **EXP-NEW** *(Mobile only)* | New Mobile | `280` | `552` | `0.9729` | `0.9180` | `0.9447` | `0.9615` | `0.7220` |
| **EXP-NEW** *(Mobile only)* | Old CCTV *(Collapse)* | `829` | `1,170` | `0.3384` | `0.1015` | `0.1562` | `0.0808` | `0.0160` |
| **EXP-NEW** *(Mobile only)* | Combined Union | `1,109` | `1,722` | `0.8317` | `0.3429` | `0.4856` | `0.3947` | `0.2512` |
| **EXP-COMBINED** *(Union)* | New Mobile | `280` | `552` | `0.9693` | `0.9051` | `0.9361` | `0.9602` | `0.7040` |
| **EXP-COMBINED** *(Union)* | Old CCTV *(Recovered)* | `829` | `1,170` | **`0.7702`** | **`0.7296`** | **`0.7493`** | **`0.7517`** | **`0.3179`** |
| **EXP-COMBINED** *(Union)* | Combined Union | `1,109` | `1,722` | **`0.8357`** | **`0.7856`** | **`0.8099`** | **`0.8437`** | **`0.4710`** |

### 1.2 Three-Seed Reproducibility Study (Phase 9: Seeds `42, 123, 2026`)
Verified from [`results/phase9_multiseed/mean_std_summary.csv`](file:///Users/sayemuddin/Desktop/research_helmet/results/phase9_multiseed/mean_std_summary.csv) (`18` total held-out evaluations across `6` Tesla T4 runs):

| Population | Metric | EXP-NEW Mean ± SD | EXP-COMBINED Mean ± SD | Mean Diff (`COMB - NEW`) | Cohen's $d$ |
|---|---|---:|---:|---:|---:|
| **New Mobile (`280`)** | Precision | `0.9737 ± 0.0019` | `0.9728 ± 0.0055` | `-0.0009` | `-0.13` |
| **New Mobile (`280`)** | Recall | `0.9180 ± 0.0069` | `0.9034 ± 0.0016` | `-0.0146` | `-1.90` |
| **New Mobile (`280`)** | F1-Score | `0.9450 ± 0.0028` | `0.9368 ± 0.0019` | `-0.0082` | `-7.45` |
| **New Mobile (`280`)** | mAP@0.50 | `0.9609 ± 0.0029` | `0.9601 ± 0.0015` | `-0.0008` | `-0.19` |
| **Old CCTV (`829`)** | Precision | `0.3709 ± 0.0627` | **`0.7773 ± 0.0073`** | **`+0.4064`** | `+7.17` |
| **Old CCTV (`829`)** | Recall | `0.1386 ± 0.0336` | **`0.7406 ± 0.0096`** | **`+0.6020`** | `+24.88` |
| **Old CCTV (`829`)** | F1-Score | `0.2009 ± 0.0433` | **`0.7585 ± 0.0084`** | **`+0.5576`** | `+15.89` |
| **Old CCTV (`829`)** | mAP@0.50 | `0.1164 ± 0.0419` | **`0.7595 ± 0.0090`** | **`+0.6431`** | `+19.55` |
| **Combined (`1,109`)** | Precision | `0.7819 ± 0.0470` | **`0.8478 ± 0.0133`** | `+0.0659` | `+1.10` |
| **Combined (`1,109`)** | Recall | `0.3698 ± 0.0242` | **`0.7835 ± 0.0085`** | **`+0.4137`** | `+16.42` |
| **Combined (`1,109`)** | F1-Score | `0.5011 ± 0.0168` | **`0.8144 ± 0.0040`** | **`+0.3132`** | `+23.55` |
| **Combined (`1,109`)** | mAP@0.50 | `0.4209 ± 0.0282` | **`0.8477 ± 0.0047`** | **`+0.4268`** | `+18.16` |

---

## 2. Phase 10 & TASK-04: Temporal Verification & Genuine Observation Track Redesign

### 2.1 Synthetic Benchmark (Phase 10: 21 Experiments)
Verified from [`results/phase10_temporal/benchmark_comparison_table.csv`](file:///Users/sayemuddin/Desktop/research_helmet/results/phase10_temporal/benchmark_comparison_table.csv): across all 3 aggregators (`simple_voting`, `confidence_weighted`, `ema`) and all 7 scenarios, Precision = `1.00`, Recall = `1.00`, False Alerts = `0`, Stability = `1.000`, Confirmation Delay = `5.0` frames (`0.167 s` at 30 fps) on violation/occlusion/crossing scenarios (`0.0 s` on helmet/compliant/marginal scenarios), with abstention rates `0.097`–`0.115`.

### 2.2 Track-Level Evidence Recovery (`TASK-04`)
Verified from [`helmet_track_level_redesign.json`](file:///Users/sayemuddin/Desktop/research_helmet/results/phase12_track_state_redesign/helmet_track_level_redesign.json) & [`event_regression.json`](file:///Users/sayemuddin/Desktop/research_helmet/results/phase12_track_state_redesign/event_regression.json) across `209` decided eligible GT tracks (including `85` $K=1$ tracks):

| Aggregator / Redesign Mode | Decided / `209` | Abstentions (`UNCERTAIN`) | Coverage | Cond. Accuracy | End-to-End Agreement | `K=1` Subgroup Agreement (`n=85`) |
|---|---:|---:|---:|---:|---:|---:|
| **Legacy `ema_final`** | `136` | `73` *(WINDOW_FLUSH_ZERO_MASS)* | `0.6507` | `0.9853` (`134/136`) | `0.6411` (`134/209`) | `0.6471` (`55/85`, `30` `UNCERTAIN`) |
| **`last_coupled`** | `209` | `0` | `1.0000` | `0.9856` (`206/209`) | `0.9856` (`206/209`) | `1.0000` (`85/85`) |
| **`majority`** | `209` | `0` | `1.0000` | `0.9904` (`207/209`) | `0.9904` (`207/209`) | `1.0000` (`85/85`) |
| **`conf_weighted` / `primary`** | **`209`** | **`0`** | **`1.0000`** | **`0.9904` (`207/209`)** | **`0.9904` (`207/209`)** | **`1.0000` (`85/85`)** |

Zero online regression across all `85` Stage J events (`events_changed = 0`).

---

## 3. Stage K Unified Audit & TASK-12 Temporal IoU Boundary Recovery

Verified from [`stagek_results.json`](file:///Users/sayemuddin/Desktop/research_helmet/results/phase12_integrated/stagek_results.json), [`bootstrap_cis.json`](file:///Users/sayemuddin/Desktop/research_helmet/results/phase12_integrated/bootstrap_cis.json), and [`task12_tiou_comparison.json`](file:///Users/sayemuddin/Desktop/research_helmet/results/phase12_task12/task12_tiou_comparison.json) (`B = 2,000` pair-cluster bootstrap replicates, `seed = 20261109`):

| Evaluation Family *(Bounded by `v1.0` Construction Circularity Where Noted)* | Point Estimate | Bootstrap / Wilson 95% CI | Details |
|---|---|---|---|
| Head detection operating point (Front / Back / Pooled) | F1 `0.9253` / `0.5145` / `0.8161` | Pooled F1 `[0.7454, 0.8762]`; P `[0.5941, 0.7796]` | Front P=`0.861`, AP50=`0.8932`; Back P=`0.3464` (`2,138` FP), AP50=`0.3467` |
| Motorcycle track-level (`gt=82, pred=100`) | P=`0.8200`, R=`1.0000`, F1=`0.9011` | — | `TP=82, FP=18, FN=0`; exact first/last/hit match `82/82` |
| Spatially valid coupled rider obs (`n=6,727`) *(circular)* | F1 `1.0000` | `[1.0000, 1.0000]` | `17` pred coupled on excluded tracks; `32` GT excluded |
| Tracking head-basis `IDF1` *(circular)* | `0.9987` (`IDP=0.9975, IDR=1.0, IDS=0`) | `[0.9970, 0.9998]` | `IDTP=6,759, IDFP=17, IDFN=0` |
| Motorcycle association (`n=10` GT linked) | P=`0.8333`, R=`1.0000`, F1=`0.9091` | Pair CI `[0.7368, 1.0000]`; Wilson P `[0.552, 0.953]` | `TP=10, FP=2` (`pair_009`, `pair_015`), `FN=0`, Abstain `38.8%` (`33/85`) |
| Orphan association (`n=1` GT linked) | F1=`1.0000` | — | `TP=1, FP=0, FN=0`, Abstain `63.3%` (`76/120`, tiny-$n$) |
| Event level (**Legacy Stage K `tIoU`**) | P=`0.7529`, R=`0.9143`, F1=`0.8258` | Pair CI `[0.7286, 0.9268]` | `TP=64, FP=21, FN=6` (`driver_no_helmet` F1=`0.8519`, `23/23`) |
| Event level (**Fixed `TASK-12` `tIoU`**) | P=`0.7765`, R=`0.9429`, F1=`0.8516` | Wilson P `[0.677, 0.852]`, R `[0.862, 0.978]` | **`TP=66 (+2), FP=19 (-2), FN=4 (-2)`**; `driver_no_helmet` F1=`0.9259`; `timing_d_mean_abs=0.0 s` |

The `+2 TP / +0.0258 F1` recovery stems deterministically from fixing degenerate zero-duration point intervals `[t, t] == [t, t]` on `pair_012` and `pair_015` inside [`compute_tiou`](file:///Users/sayemuddin/Desktop/research_helmet/scripts/reproducibility/evaluate.py#L33).

---

## 4. GAP-B & TASK-10: Keyframe Retention, Coordinate Scaling & Real-Video Cue C

### 4.1 GAP-B Offline Replay (`TASK-06/06A`, [`gapb_evaluation_results.json`](file:///Users/sayemuddin/Desktop/research_helmet/results/phase12_gapb_redesign/gapb_evaluation_results.json))
- **Keyframe Retention (`cap <= 24/stream`):** Stage J stored descriptors for only `7 / 100` (`7%`) motorcycle tracklets (`56 EMPTY_CROP`, `37 NO_MATCHING_FRAME_INDEX`), and naive stride-based retention reached `49 / 100` (`63 / 100` after `TASK-06A` simulator fix). Protected-anchor retention achieved **`98 / 100`** initially (`100 / 100` in the `TASK-06A` synchronized simulator, `0` stale refs, `<= 24` keyframes on all `50/50` streams) and recovered **`9 / 9`** real PNG failure crops under explicit `4K -> 720p` coordinate scaling (`9 / 9` rejected when `native_wh=None`).
- **Border Classification (`GAP-B2`):** Replaced legacy `1920x1080` assumptions with true stream dimensions (`3840x2160`), eliminating **`51` entry FP** and **`42` exit FP** (`86 -> 35` entry, `75 -> 33` exit).
- **Evidence Preservation (`GAP-B3`):** Recovered `74 / 74` rider subtracks (`11 -> 66 NO_HELMET`, `1 -> 20 HELMET`) and `29 / 29` orphan tracks from legacy `UNCERTAIN`, with `0` violations of the `min_active_cues_required = 2` gate.

### 4.2 TASK-10 Bounded Real-Video Cue C Verification ([`task10_cue_c_results.json`](file:///Users/sayemuddin/Desktop/research_helmet/results/phase12_task10/task10_cue_c_results.json))
Executed in `17.279 s` (`1,806.03 MB` peak RSS) on decoded `720p` (`pair_001, 013, 014, 015, 019`) and `4K` (`pair_005`) streams (`55` comparable entities):
- **Descriptor Availability:** Rose from `6 / 75` (`8.0%`) in pre-GAP-B Stage J to **`55 / 55` (`100.0%`)** in both `C2_A4_720p_kf` and `C3_A4_4k_native` (and `75 / 75` in `C1_A0_720p_kf`), while unknown-dimensions probing (`C4`) caught `COORDINATE_SPACE_MISMATCH_UNKNOWN_NATIVE_DIMENSIONS` cleanly (`14.29%` moto availability).
- **Resolution Invariance:** Mean `720p vs 4K` composite self-similarity = **`0.9971`** (min `0.9857`); mean `|Cue C_720p - Cue C_4K|` = **`0.0083`** (max `0.0278`).
- **Opposing-View Discrimination & Knife-Edge Finding:** Positive pairs (`n=5`) averaged `Cue C = 0.7223` (median `0.7426`) vs. negative pairs (`n=20`) at `0.6641` (median `0.7306`). This overlap pushed borderline `pair_005` from `0.6192` (`UNMATCHED`) to `0.6197` (`C2` `720p`, `UNMATCHED`) and **`0.6203` (`C3` `4K`, false `LINKED`, `+0.0003` over `0.62`)** — proving raw histograms are invariant across resolutions but weakly discriminative alone.

---

## 5. GAP-C (`TASK-09/09A/09B`): Motorcycle-Anchored Kinematics, Consensus & Orphan Lifecycle

Verified from [`gapc_ablation_summary.csv`](file:///Users/sayemuddin/Desktop/research_helmet/results/phase12_gapc/gapc_ablation_summary.csv) across all `25` video pairs (`100` motorcycle tracks, `9,244` coupled heads):

| Mode | Front Coupled Ret. | Back Coupled Ret. | Combined Ret. | Abstained Updates | Subtracks $K=1$ Multi-Frame Elig. | Out-of-Bounds Boxes | Role Mismatches (`multi_driver_motos`) | Exported Orphans (`1-hit` `<0.45`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`A0` Legacy** | `84.81%` (`5,297/6,246`) | **`28.69%` (`860/2,998`)** | `66.61%` | `2,943` | `30` | `660` | `0` | `130` (`56` / `34`) |
| **`A1` +RelKin** | `90.17%` | `70.51%` | `83.79%` | `1,364` | `19` | `0` | `1` | `128` (`54` / `34`) |
| **`A2` +Unlock/Coast** | `95.89%` | `91.66%` | `94.52%` | `229` | `1` | `0` | `47` (`13`) | `128` (`54` / `34`) |
| **`A3` +RoleConsensus** | `95.87%` | `91.66%` | `94.50%` | `230` | `1` | `0` | `0` (`0`) | `128` (`54` / `34`) |
| **`A4` Full GAP-C** | **`95.87%`** | **`91.66%` (`2,748/2,998`)** | **`94.50%` (`8,736`)** | **`230` (`-92.2%`)** | **`1`** | **`0`** | **`0` (`0`)** | **`68` (`0` / `0`)** |

---

## 6. TASK-11: Dual-Camera Bayesian Log-Odds Fusion & Single-vs-Dual Ablation

Executed in `5.304 s` (`458.92 MB` peak RSS) across `50` streams and `70` physical passes (`40` violation passes, `46` decided passes), with **Wilson 95% score CIs** verified over [`task11_ablation_summary.csv`](file:///Users/sayemuddin/Desktop/research_helmet/results/phase12_task11/task11_ablation_summary.csv):

| Arm (`Tracker` + `Fusion`) | Strict F1 *(Fixed tIoU)* | Strict Precision `[95% CI]` | Strict Recall `[95% CI]` | Phys. Pass Coverage `[95% CI]` | Viol. Binary Recall (`k/40`) `[95% CI]` | Decided Pass Acc. (`k/46`) `[95% CI]` | Abstain / Quar. |
|---|---:|---|---|---|---|---|---|
| **`A0_FRONT_ONLY`** | `0.4746` | `0.5833` `[0.443, 0.712]` | `0.4000` `[0.293, 0.517]` | `0.6000` (`42/70`) `[0.483, 0.707]` | `0.6000` (`24/40`) `[0.446, 0.737]` | `0.5000` (`23/46`) `[0.361, 0.639]` | `0 / 0` |
| **`A0_BACK_ONLY`** | `0.3443` | `0.4038` `[0.282, 0.539]` | `0.3000` `[0.205, 0.415]` | `0.5714` (`40/70`) `[0.455, 0.681]` | `0.3500` (`14/40`) `[0.221, 0.505]` | `0.3261` (`15/46`) `[0.209, 0.470]` | `0 / 0` |
| **`A0_DUAL_LEGACY`** | **`0.8182`** | `0.7500` `[0.648, 0.830]` | **`0.9000`** `[0.808, 0.951]` | `0.9571` (`67/70`) `[0.881, 0.985]` | **`0.9250`** (`37/40`) `[0.801, 0.974]` | **`0.8913`** (`41/46`) `[0.770, 0.953]` | `0 / 0` |
| **`A0_DUAL_FUSED`** | `0.6753` | `0.6190` `[0.512, 0.716]` | `0.7429` `[0.630, 0.831]` | `0.9571` (`67/70`) `[0.881, 0.985]` | `0.6750` (`27/40`) `[0.520, 0.799]` | `0.6522` (`30/46`) `[0.508, 0.773]` | `1 / 4` |
| **`A4_FRONT_ONLY`** | `0.4915` | `0.6042` `[0.463, 0.730]` | `0.4143` `[0.306, 0.531]` | `0.6000` (`42/70`) `[0.483, 0.707]` | `0.6000` (`24/40`) `[0.446, 0.737]` | `0.5217` (`24/46`) `[0.381, 0.659]` | `0 / 0` |
| **`A4_BACK_ONLY`** | `0.3115` | `0.3654` `[0.248, 0.501]` | `0.2714` `[0.181, 0.385]` | `0.5714` (`40/70`) `[0.455, 0.681]` | `0.4500` (`18/40`) `[0.307, 0.602]` | `0.2609` (`12/46`) `[0.156, 0.403]` | `0 / 0` |
| **`A4_DUAL_LEGACY`** | `0.6494` | `0.5952` `[0.488, 0.694]` | `0.7143` `[0.599, 0.807]` | `0.9571` (`67/70`) `[0.881, 0.985]` | `0.8000` (`32/40`) `[0.652, 0.895]` | `0.6304` (`29/46`) `[0.486, 0.755]` | `0 / 0` |
| **`A4_DUAL_FUSED`** | `0.6494` | `0.5952` `[0.488, 0.694]` | `0.7143` `[0.599, 0.807]` | `0.9571` (`67/70`) `[0.881, 0.985]` | `0.7250` (`29/40`) `[0.572, 0.839]` | `0.6087` (`28/46`) `[0.465, 0.736]` | `1 / 4` |
| **`A4_DUAL_FUSED_SPLIT`** | **`0.6582`** | `0.5909` `[0.486, 0.688]` | **`0.7429`** `[0.630, 0.831]` | **`0.9714` (`68/70`)** `[0.902, 0.992]` | **`0.8250` (`33/40`)** `[0.681, 0.913]` | **`0.6522` (`30/46`)** `[0.508, 0.773]` | `1 / 8` |
| **`A4_DUAL_FUSED_MINOBS2`** | `0.6234` | `0.5714` `[0.465, 0.672]` | `0.6857` `[0.570, 0.782]` | `0.9571` (`67/70`) `[0.881, 0.985]` | `0.6750` (`27/40`) `[0.520, 0.799]` | `0.5652` (`26/46`) `[0.422, 0.698]` | `1 / 4` |

### Exact Paired McNemar Tests (Two-Sided Exact Binomial)
- **`A4_DUAL_FUSED_SPLIT` vs. `A4_FRONT_ONLY`:** physical coverage `+27 / -1` (`p = 2.16e-07`), binary violation recall `+10 / -1` (`p = 0.0117`), strict schema TP `+23 / -0` (`p = 2.38e-07`).
- **`A4_DUAL_FUSED_SPLIT` vs. `A4_BACK_ONLY`:** physical coverage `+29 / -1` (`p = 5.77e-08`), binary violation recall `+15 / -0` (`p = 6.10e-05`), decided accuracy `+20 / -2` (`p = 1.21e-04`), strict TP `+33 / -0` (`p = 2.33e-10`).
- **Forensic Resolutions & `A4` vs `A0` Strict Divergence:** `A4_DUAL_FUSED` resolves all `3/3` Stage J $K=1$ vetos (`pair_012: P=0.2084`, `pair_018: P=0.9137`, `pair_024: P=0.9560`), quarantines false `pair_015` conflict (`K=0.35`), splits `pair_020` bicycle (`DRIVER_NO_HELMET` front + `INSUFFICIENT_VISUAL_EVIDENCE` back), links `pair_009` orphan (`P=0.7307`), and rescues `A4_BACK_ONLY` binary violation recall from `35.0% -> 45.0%`. On `v1.0` strict schema, `A4_DUAL_LEGACY` (`50/70`) trails `A0_DUAL_LEGACY` (`63/70`; `+1 / -14`, `p = 0.00098`) strictly because `v1.0` was labeled relative to pre-GAP-C `A0` subtracks whereas `A4` recovers previously dropped rear pillions and updates longitudinal slot roles.
