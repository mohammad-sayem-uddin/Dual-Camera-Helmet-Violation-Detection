# ARGON TASK-11 ARCHITECTURE DESIGN — Confidence-Aware Dual-Camera Evidence Fusion (`src/fusion/`) & Fair Single-vs-Dual Ablation

**Date:** 2026-11-12  
**Architecture Authority:** Gemini Argon  
**Prerequisite Cleared Gates:**
- **Gate A (`TASK-09A` / `TASK-09B`, commit `8cc3c96`):** Back-camera coupled head retention restored (`28.7% -> 91.7%`), `abstained_updates` cut `92.2%`, longitudinal role consensus enabled, `0` zero-driver motos (`no_driver_motos = 0`), `18/18` checks & `184/184` offline pytest pass.
- **Gate B (`TASK-10`, commit `00bd15f`):** Bounded real-video `Cue C` validated on `8/50` `.mp4` streams (`55/55` = `100%` descriptor recovery, `1280x720` vs `3840x2160` self-similarity `mean = 0.9971`, uncalibrated opposing-view overlap floor `~0.55–0.85` documented).

---

## 1. Architectural Problem Statement

Prior to `TASK-11`, `src/fusion/` did not exist. Per `ARGON_TASK08_NEXT_RESEARCH_GAP_ASSESSMENT.md` (§6.2), event derivation inside [`derive_pair_events()`](file:///Users/sayemuddin/Desktop/research_helmet/src/pipeline/dual_camera_pipeline.py#L1231-L1345) suffered from four structural gaps:

1. **Equal-Weight Set-Equality Veto (`K=1` Override):** [`_merged_riders()`](file:///Users/sayemuddin/Desktop/research_helmet/src/pipeline/dual_camera_pipeline.py#L1138-L1198) collapsed `conf = {s for ...}` so a single noisy opposing-camera frame (`K = 1`) vetoed a dense `100+`-frame track into `"UNCERTAIN"`, while ignoring observation counts, confidence mass, crop scale, temporal flip noise, viewpoint asymmetry, and synchronization residual bounds.
2. **Disconnected Cross-Camera Sub-Entities:** [`derive_pair_events()`](file:///Users/sayemuddin/Desktop/research_helmet/src/pipeline/dual_camera_pipeline.py#L1231-L1345) ignored [`RiderAssociationResult`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/tracklet_models.py#L615-L656) (pairing front/back riders only by slot integer/label heuristics) and ignored [`OrphanAssociationResult`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/tracklet_models.py#L704-L741) entirely.
3. **Unquarantined `AMBIGUOUS_ABSTAIN` Links:** Motorcycle links flagged `AMBIGUOUS_ABSTAIN` (6 Stage J events, e.g. `pair_020` linking `front_moto_0001` with excluded bicycle `back_moto_0003`) were still cross-view merged inside `_merged_riders()`.
4. **No Standardized Fair Single-vs-Dual Ablation Harness:** Single-camera baselines (`FRONT_ONLY`, `BACK_ONLY`) cannot match opposite-camera keys under exact `(pair, front_track_id, back_track_id)` string tuple scoring; a fair ablation must score **both** exact dual-schema parity and **view-projected physical-pass metrics** across all `70` GT events and `209` decided rider tracks, on both pre-GAP-C (`A0`) and post-GAP-C (`A4`) tracking.

---

## 2. Module Boundaries (`src/fusion/`)

Two cohesive, single-purpose modules will be created under `src/fusion/` without altering existing `src/temporal/` or breaking legacy Stage J parity:

1. [`src/fusion/evidence_fusion.py`](file:///Users/sayemuddin/Desktop/research_helmet/src/fusion/evidence_fusion.py) — Mathematical primitives & typed evidence structures:
   - `FusionDecisionState` (`VERIFIED_DUAL_VIOLATION`, `VERIFIED_DUAL_COMPLIANT`, `SINGLE_VIEW_VIOLATION_FRONT`, `SINGLE_VIEW_VIOLATION_BACK`, `SINGLE_VIEW_COMPLIANT_FRONT`, `SINGLE_VIEW_COMPLIANT_BACK`, `EVIDENCE_CONFLICT_ABSTAIN`, `AMBIGUOUS_ASSOC_ABSTAIN`, `LOW_CONFIDENCE_ABSTAIN`).
   - `FusionConfig` (all weights, Beta prior, thresholds, sync decay, and quarantine policy).
   - `ViewRiderEvidence` (unified normalized input constructed from live `RiderSubTrack`/`OrphanRiderTrack` or read-only snapshot dicts from [`RiderAssociationResult`](file:///Users/sayemuddin/Desktop/research_helmet/src/association/tracklet_models.py#L615-L656)).
   - Pure scoring functions: `compute_view_posterior_and_weight()`, `compute_sync_reliability()`, `compute_contradiction_metric()`, `fuse_rider_beliefs()`, `evaluate_single_view_belief()`.
2. [`src/fusion/dual_camera_fusion.py`](file:///Users/sayemuddin/Desktop/research_helmet/src/fusion/dual_camera_fusion.py) — Track- and pair-level orchestrator (`DualCameraFusionEngine`):
   - Fuses `LINKED` motorcycles using `RiderAssociationResult` first, preserving unmatched single-view slots without cross-view contamination.
   - Quarantines `AMBIGUOUS_ABSTAIN` motorcycle links (never pooling Front and Back posteriors or licensing cross-view plates), configurable between `quarantine_split` (separate single-view events) and `quarantine_preserve_keys` (schema-preserving abstention).
   - Quarantines `AMBIGUOUS_ABSTAIN` rider pairs within `LINKED` motorcycles rather than forcing a single cross-camera state.
   - Evaluates `OrphanAssociationResult` as first-class `OrphanFusionResult` records (`implies_confirmed_motorcycle_link = False`).
   - Projects pipelines in single-camera modes (`front_only`, `back_only`) as well as dual-camera mode (`dual_fused`).

---

## 3. Formal Mathematical Specification

### 3.1 Beta-Smoothed Per-Camera Posterior

For view $c \in \{\text{front}, \text{back}\}$ and rider entity $r$ with genuine coupled observation mass $(w_{\text{nh}}, w_{\text{h}})$ (using accumulated detection confidence sums `coupled_confidence_weights` when positive, else genuine counts `(no_helmet_count, helmet_count)`) and symmetric uniform prior $\alpha_0 = \beta_0 = 1.0$:

$$P_{c,r}(\text{no\_helmet}) = \frac{w_{\text{nh}} + \alpha_0}{w_{\text{nh}} + w_{\text{h}} + \alpha_0 + \beta_0} \in (0, 1)$$

Clamped to $[\varepsilon, 1 - \varepsilon]$ ($\varepsilon = 10^{-4}$) for log-odds stability:

$$L_{c,r} = \ln\frac{P_{c,r}}{1 - P_{c,r}}$$

**Key theoretical property:** With $\alpha_0 = \beta_0 = 1.0$, a `K = 1` opposing observation has maximum posterior $P \le \frac{1 + 1}{1 + 2} = 0.6667$, yielding maximum conflict contribution $2|P - 0.5| \le 0.3333 < \tau_{\text{conflict}} = 0.35$. Thus a solitary `K = 1` noise glitch **never triggers a false conflict deadlock**, whereas $K \ge 2$ unanimous opposing evidence ($P \ge \frac{2+1}{2+2}$ at conf $=1$ or $\ge 0.68$ at moderate conf) can trigger `EVIDENCE_CONFLICT_ABSTAIN` when opposed by a strong opposite-view track.

### 3.2 Observation Quality, Stability & Viewpoint Reliability Weight

Each view's log-odds weight combines four interpretable factors:

$$W_{c,r} = \underbrace{\frac{N_{c,r}}{N_{c,r} + N_{\text{half}}}}_{\text{sample maturity}} \cdot \underbrace{q_{\text{scale}}(A_{\text{head}}) \cdot (1 - \lambda_b f_{\text{border}})}_{\text{spatial crop quality}} \cdot \underbrace{\max(0.5,\, 1 - \lambda_{\text{flip}} r_{\text{flip}})}_{\text{temporal stability}} \cdot \gamma_c \cdot m_{\text{fsm}}$$

where:
- $N_{\text{half}} = 4.0$ yields $0.20$ at $N=1$, $0.50$ at $N=4$, $0.83$ at $N=20$, $0.97$ at $N=150$.
- $q_{\text{scale}} = \text{clip}(A_{\text{head}} / 1600\,\text{px}^2,\, 0.25,\, 1.0)$ when head bbox area is available, else $1.0$; $f_{\text{border}}$ is border-clipping fraction ($\lambda_b = 0.35$).
- $r_{\text{flip}} = S_{\text{switches}} / \max(1, N_{c,r} - 1)$ with $\lambda_{\text{flip}} = 0.50$.
- $\gamma_{\text{front}} = 1.10$, $\gamma_{\text{back}} = 0.95$ (anterior face visibility vs posterior occipital shell).
- $m_{\text{fsm}} = 1.08$ when Phase 10 confirms (`NO_HELMET_CONFIRMED` / `HELMET_CONFIRMED` or `alerts_count > 0`), else $1.00$.

### 3.3 Synchronization Residual Discount & Cross-View Log-Odds Fusion

Per `PHASE12B` §8.2:

$$\phi_{\text{sync}}(\epsilon_{\text{sync}}, \bar{\delta}_{\text{pts}}) = \max\!\left(0.80,\, \exp\!\left(-0.003 \cdot \max(0,\, \epsilon_{\text{sync}} + \bar{\delta}_{\text{pts}} - 16.67)\right)\right)$$

Let $C_{\text{assoc}} = \text{clip}(S_{\text{assoc}}, 0.05, 1.0) \cdot \phi_{\text{sync}}$. When both Front and Back observations are available and neither is quarantined:

$$L_{\text{fused}} = \frac{W_F L_F + C_{\text{assoc}} W_B L_B}{W_F + C_{\text{assoc}} W_B} + \beta_{\text{corrob}} \cdot \mathbb{I}[L_F L_B > 0] \cdot C_{\text{assoc}} \cdot \min(|L_F|, |L_B|)$$

with $\beta_{\text{corrob}} = 0.20$ and posterior $P_{\text{fused}} = \sigma(L_{\text{fused}})$.

### 3.4 Contradiction & Low-Confidence Decision Gates

Contradiction metric per `PHASE12B` §9:

$$K_{\text{conflict}}(F, B) = \mathbb{I}[(P_F - 0.5)(P_B - 0.5) < 0] \cdot 2\min(|P_F - 0.5|, |P_B - 0.5|) \in [0, 1]$$

Decision mapping on decisive thresholds ($\tau_{\text{viol}} = 0.58$, $\tau_{\text{comp}} = 0.42$, $\tau_{\text{conflict}} = 0.35$):
1. Cross-view link `AMBIGUOUS_ABSTAIN` $\implies$ **`AMBIGUOUS_ASSOC_ABSTAIN`** (quarantined; views never pooled together, cross-view plate linking forbidden).
2. Dual matched & $K_{\text{conflict}} \ge 0.35$ $\implies$ **`EVIDENCE_CONFLICT_ABSTAIN`** (`helmet_state = "UNCERTAIN"`).
3. Matched or single view & $P \ge \tau_{\text{viol}}$ (and $N \ge N_{\text{min}}$) $\implies$ `NO_HELMET` (`VERIFIED_DUAL_VIOLATION` or `SINGLE_VIEW_VIOLATION_*`).
4. Matched or single view & $P \le \tau_{\text{comp}}$ (and $N \ge N_{\text{min}}$) $\implies$ `HELMET` (`VERIFIED_DUAL_COMPLIANT` or `SINGLE_VIEW_COMPLIANT_*`).
5. Otherwise ($\tau_{\text{comp}} < P < \tau_{\text{viol}}$ or $N < N_{\text{min}}$) $\implies$ **`LOW_CONFIDENCE_ABSTAIN`** (`helmet_state = "UNCERTAIN"`).

---

## 4. Non-Destructive Integration & Preserved Guarantees

- [`PipelineConfig`](file:///Users/sayemuddin/Desktop/research_helmet/src/pipeline/dual_camera_pipeline.py#L173-L188) gains `fusion_mode: str = "legacy"` (`"legacy"` | `"dual_fused"` | `"front_only"` | `"back_only"`) and optional `fusion_config` (stored in a small defaultable field).
- [`derive_pair_events()`](file:///Users/sayemuddin/Desktop/research_helmet/src/pipeline/dual_camera_pipeline.py#L1231-L1345) keeps `fusion_mode: str = "legacy"` as default so all historical Stage J / TASK-04 parity checks (`event_regression_task04.py`: `85/85 changed=0`) remain **100% bit-exact unaltered**.

---

## 5. Fair Single-vs-Dual Ablation Design (`scripts/reproducibility/evaluate_task11_fusion.py`)

To avoid circular or camera-biased comparisons, `evaluate_task11_fusion.py` replays all `50` cached Stage J detection streams across all `25` pairs without touching frozen artifacts and evaluates:

1. **Mode Matrix:** `FRONT_ONLY`, `BACK_ONLY`, `DUAL_LEGACY`, `DUAL_FUSED` (quarantined), evaluated on **both** Stage J baseline tracking (`A0`) and GAP-C tracking (`A4`) so the confounding effect of the Stage J Back-camera bug is quantified directly.
2. **Evaluation Lenses:**
   - **Lens A (Strict GT-70 Schema):** Exact `(pair, front_track_id, back_track_id)` event type + temporal IoU $\ge 0.5$ matching for `DUAL_LEGACY` vs `DUAL_FUSED` (`quarantine_preserve_keys` and `quarantine_split`).
   - **Lens B (Fair View-Projected Physical Pass Benchmark):** Evaluates every single-camera and dual-camera system on all `70` GT physical motorcycle passes (`30` front-only, `28` back-only, `12` dual-visible) by matching a predicted event whenever its `front_track_id` OR `back_track_id` belongs to that physical pass (with `DUAL_FUSED` credited once per linked physical pass so no double-counting occurs!). Also measures decided-violation recall over all `40` GT violation passes and multi-rider completeness.
   - **Lens C (Forensic Case Checks):** Verifies `K=1` noise immunity (`pair_012`, `pair_018`, `pair_024` Stage J scenarios), false-link conflict quarantine (`pair_015`), cross-slot `RiderAssociation` pairing (`pair_019`, `pair_021`), `AMBIGUOUS_ABSTAIN` quarantine (`pair_007`, `pair_013`, `pair_020`), and first-class orphan fusion outcomes.
