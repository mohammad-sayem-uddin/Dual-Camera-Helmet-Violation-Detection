"""Confidence-aware per-view and cross-camera rider helmet belief fusion.

Implements TASK-11 (`src/fusion/`) evidence quality weighting, Beta-Binomial
posterior shrinkage (eliminating K=1 opposing-view vetoes), synchronization
residual discounting, log-odds fusion with corroboration boost, and explicit
contradiction (`EVIDENCE_CONFLICT_ABSTAIN`), ambiguous-link, and low-confidence
abstention gates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any


_HELMET_DECIDED = ("HELMET", "NO_HELMET")
_CONFIRMED_STATES = ("HELMET_CONFIRMED", "NO_HELMET_CONFIRMED")


class FusionDecisionState(str, Enum):
    """Canonical decision state taxonomy for fused and single-view rider evidence."""

    VERIFIED_DUAL_VIOLATION = "VERIFIED_DUAL_VIOLATION"
    VERIFIED_DUAL_COMPLIANT = "VERIFIED_DUAL_COMPLIANT"
    SINGLE_VIEW_VIOLATION_FRONT = "SINGLE_VIEW_VIOLATION_FRONT"
    SINGLE_VIEW_VIOLATION_BACK = "SINGLE_VIEW_VIOLATION_BACK"
    SINGLE_VIEW_COMPLIANT_FRONT = "SINGLE_VIEW_COMPLIANT_FRONT"
    SINGLE_VIEW_COMPLIANT_BACK = "SINGLE_VIEW_COMPLIANT_BACK"
    EVIDENCE_CONFLICT_ABSTAIN = "EVIDENCE_CONFLICT_ABSTAIN"
    AMBIGUOUS_ASSOC_ABSTAIN = "AMBIGUOUS_ASSOC_ABSTAIN"
    LOW_CONFIDENCE_ABSTAIN = "LOW_CONFIDENCE_ABSTAIN"


@dataclass(frozen=True)
class FusionConfig:
    """Hyperparameters governing dual-camera log-odds belief fusion and abstention."""

    prior_pseudo_count: float = 1.0  # Symmetric Beta(1, 1) prevents K=1 veto spikes
    sample_maturity_half_count: float = 4.0
    head_area_reference_px2: float = 1600.0
    min_scale_quality: float = 0.25
    border_penalty_weight: float = 0.35
    flip_penalty_weight: float = 0.50
    min_stability_quality: float = 0.50
    weight_front_viewpoint: float = 1.10
    weight_back_viewpoint: float = 0.95
    fsm_confirmation_multiplier: float = 1.08
    sync_threshold_ms: float = 16.67
    sync_decay_per_ms: float = 0.003
    min_sync_reliability: float = 0.80
    corroboration_beta: float = 0.20
    conflict_threshold: float = 0.35
    violation_prob_threshold: float = 0.58
    compliant_prob_threshold: float = 0.42
    min_observations_for_decision: int = 1
    quarantine_ambiguous_links: bool = True
    split_ambiguous_moto_to_single_view_events: bool = False
    ambiguous_link_abstain_citation: bool = True
    prob_clamp_eps: float = 1e-4


@dataclass(frozen=True)
class ViewRiderEvidence:
    """Normalized per-camera rider or orphan helmet evidence summary."""

    camera_id: str  # "front" or "back"
    track_id: str
    slot_index: int | None = None
    slot_role: str = "uncertain"
    consensus_slot_role: str | None = None
    ordering_confident: bool = True
    helmet_count: int = 0
    no_helmet_count: int = 0
    helmet_conf_sum: float = 0.0
    no_helmet_conf_sum: float = 0.0
    state_switches: int = 0
    track_helmet_state: str = "UNCERTAIN"
    phase10_temporal_state: str = "UNCERTAIN"
    phase10_alerts_count: int = 0
    alert_ids: tuple[str, ...] = ()
    mean_head_area_px: float | None = None
    near_border_fraction: float = 0.0
    t_start_ref_sec: float | None = None
    t_end_ref_sec: float | None = None

    @property
    def total_observations(self) -> int:
        return int(self.helmet_count) + int(self.no_helmet_count)

    @property
    def effective_role(self) -> str:
        role = str(self.consensus_slot_role or self.slot_role or "").strip()
        return role or "uncertain"

    @classmethod
    def from_phase10_readonly(
        cls,
        *,
        camera_id: str,
        subtrack_id: str,
        slot_role: str,
        readonly_dict: dict[str, Any] | None,
        slot_index: int | None = None,
        consensus_slot_role: str | None = None,
        ordering_confident: bool = True,
        alert_ids: tuple[str, ...] = (),
        mean_head_area_px: float | None = None,
        near_border_fraction: float = 0.0,
        t_start_ref_sec: float | None = None,
        t_end_ref_sec: float | None = None,
    ) -> ViewRiderEvidence:
        """Build normalized evidence from a track record or `front/back_phase10_readonly`."""
        ro = readonly_dict or {}
        ge = ro.get("genuine_observation_helmet_evidence") or ro
        cw = ge.get("coupled_confidence_weights") or {}
        h_cnt = int(ge.get("coupled_helmet_count", 0))
        nh_cnt = int(ge.get("coupled_no_helmet_count", 0))
        h_sum = float(cw.get("helmet", 0.0))
        nh_sum = float(cw.get("no_helmet", 0.0))
        return cls(
            camera_id=str(camera_id),
            track_id=str(subtrack_id),
            slot_index=slot_index,
            slot_role=str(slot_role or "uncertain"),
            consensus_slot_role=consensus_slot_role or ro.get("consensus_slot_role"),
            ordering_confident=bool(ordering_confident),
            helmet_count=h_cnt,
            no_helmet_count=nh_cnt,
            helmet_conf_sum=h_sum,
            no_helmet_conf_sum=nh_sum,
            state_switches=int(ge.get("coupled_state_switches", 0)),
            track_helmet_state=str(ge.get("track_helmet_state", "UNCERTAIN")),
            phase10_temporal_state=str(ro.get("temporal_state", "UNCERTAIN")),
            phase10_alerts_count=int(ro.get("alerts_count", len(alert_ids))),
            alert_ids=tuple(alert_ids),
            mean_head_area_px=mean_head_area_px,
            near_border_fraction=float(near_border_fraction),
            t_start_ref_sec=t_start_ref_sec,
            t_end_ref_sec=t_end_ref_sec,
        )


@dataclass(frozen=True)
class ViewPosteriorSummary:
    """Computed Bayesian posterior, log-odds, and reliability weight for one view."""

    camera_id: str
    posterior_no_helmet: float
    log_odds_no_helmet: float
    quality_weight: float
    sample_maturity: float
    scale_quality: float
    stability_quality: float
    observations: int


@dataclass(frozen=True)
class RiderFusionResult:
    """Result of fusing (or single-view scoring) one physical rider entity."""

    canonical_slot: int
    role: str
    helmet_state: str
    provenance: str
    fusion_decision_state: FusionDecisionState
    fused_p_no_helmet: float
    fused_log_odds: float
    contradiction_k: float
    sync_reliability: float
    association_score: float | None
    cross_view_fused: bool
    quarantined: bool
    abstained: bool
    decision_reasons: tuple[str, ...] = ()
    violation_window_ref_sec: tuple[float, float] | None = None
    front_evidence: ViewRiderEvidence | None = None
    back_evidence: ViewRiderEvidence | None = None
    front_summary: ViewPosteriorSummary | None = None
    back_summary: ViewPosteriorSummary | None = None
    raw_subtrack_slots: tuple[int, ...] = ()
    recorded_slot_roles: tuple[str, ...] = ()
    alert_ids: tuple[str, ...] = ()

    def to_event_rider_dict(self) -> dict[str, Any]:
        """Serialize to the Stage J `events.json[*].riders[*]` contract plus fusion audit fields."""
        per_view: dict[str, str] = {}
        if self.front_evidence is not None:
            per_view["front"] = self.front_evidence.track_helmet_state
        if self.back_evidence is not None:
            per_view["back"] = self.back_evidence.track_helmet_state
        return {
            "rider_slot": int(self.canonical_slot),
            "raw_subtrack_slots": list(self.raw_subtrack_slots),
            "role": self.role,
            "recorded_slot_roles": list(self.recorded_slot_roles),
            "helmet_state": self.helmet_state,
            "helmet_state_provenance": self.provenance,
            "violation_window_ref_sec": (
                list(self.violation_window_ref_sec)
                if self.violation_window_ref_sec is not None
                else None
            ),
            "fusion_decision_state": self.fusion_decision_state.value,
            "fused_p_no_helmet": round(float(self.fused_p_no_helmet), 4),
            "contradiction_k": round(float(self.contradiction_k), 4),
            "sync_reliability": round(float(self.sync_reliability), 4),
            "cross_view_fused": bool(self.cross_view_fused),
            "quarantined": bool(self.quarantined),
            "decision_reasons": list(self.decision_reasons),
            "per_view_track_states": per_view,
        }


def compute_mean_head_area_from_history(
    pts_history: list[tuple[Any, ...]] | tuple[tuple[Any, ...], ...],
) -> float | None:
    """Compute mean head bounding-box area (px^2) from coupled history tuples."""
    areas: list[float] = []
    for entry in pts_history:
        if len(entry) < 3:
            continue
        bbox = entry[2]
        if len(bbox) != 4:
            continue
        w = max(0.0, float(bbox[2]) - float(bbox[0]))
        h = max(0.0, float(bbox[3]) - float(bbox[1]))
        if w > 0.0 and h > 0.0:
            areas.append(w * h)
    if not areas:
        return None
    return float(sum(areas) / len(areas))


def compute_sync_reliability(
    epsilon_sync_ms: float = 0.0,
    mean_delta_pts_ms: float = 0.0,
    config: FusionConfig | None = None,
) -> float:
    """Synchronization residual discount factor phi_sync in [min_sync, 1.0]."""
    cfg = config or FusionConfig()
    excess_ms = max(
        0.0,
        float(epsilon_sync_ms) + float(mean_delta_pts_ms) - cfg.sync_threshold_ms,
    )
    return float(
        max(
            cfg.min_sync_reliability,
            min(1.0, math.exp(-cfg.sync_decay_per_ms * excess_ms)),
        )
    )


def compute_contradiction_metric(p_front: float, p_back: float) -> float:
    """Semantic conflict metric K_conflict in [0, 1] between opposing posteriors."""
    df = float(p_front) - 0.5
    db = float(p_back) - 0.5
    if df * db >= 0.0:
        return 0.0
    return float(min(1.0, 2.0 * min(abs(df), abs(db))))


def _sigmoid(logit: float) -> float:
    if logit >= 0.0:
        z = math.exp(-logit)
        return 1.0 / (1.0 + z)
    z = math.exp(logit)
    return z / (1.0 + z)


def _clamp_prob(p: float, eps: float) -> float:
    return max(eps, min(1.0 - eps, float(p)))


def compute_view_posterior_and_weight(
    ev: ViewRiderEvidence,
    config: FusionConfig | None = None,
) -> ViewPosteriorSummary:
    """Compute Beta-smoothed per-view posterior, log-odds, and reliability weight."""
    cfg = config or FusionConfig()
    n_obs = ev.total_observations
    mass_total = float(ev.helmet_conf_sum) + float(ev.no_helmet_conf_sum)
    if mass_total > 1e-8:
        w_nh = max(0.0, float(ev.no_helmet_conf_sum))
        w_h = max(0.0, float(ev.helmet_conf_sum))
    else:
        w_nh = max(0.0, float(ev.no_helmet_count))
        w_h = max(0.0, float(ev.helmet_count))

    alpha0 = float(cfg.prior_pseudo_count)
    denom = w_nh + w_h + (2.0 * alpha0)
    p_nh = _clamp_prob((w_nh + alpha0) / denom if denom > 0.0 else 0.5, cfg.prob_clamp_eps)
    log_odds = math.log(p_nh / (1.0 - p_nh))

    if n_obs <= 0:
        return ViewPosteriorSummary(
            camera_id=ev.camera_id,
            posterior_no_helmet=0.5,
            log_odds_no_helmet=0.0,
            quality_weight=0.0,
            sample_maturity=0.0,
            scale_quality=1.0,
            stability_quality=1.0,
            observations=0,
        )

    sample_maturity = float(n_obs) / (float(n_obs) + float(cfg.sample_maturity_half_count))
    if ev.mean_head_area_px is not None and cfg.head_area_reference_px2 > 0.0:
        raw_scale = float(ev.mean_head_area_px) / float(cfg.head_area_reference_px2)
        scale_q = max(cfg.min_scale_quality, min(1.0, raw_scale))
    else:
        scale_q = 1.0
    border_mult = max(0.25, 1.0 - cfg.border_penalty_weight * max(0.0, min(1.0, ev.near_border_fraction)))
    crop_q = scale_q * border_mult

    flip_rate = float(max(0, ev.state_switches)) / float(max(1, n_obs - 1))
    stability_q = max(cfg.min_stability_quality, 1.0 - cfg.flip_penalty_weight * flip_rate)

    gamma = (
        cfg.weight_front_viewpoint
        if ev.camera_id == "front"
        else cfg.weight_back_viewpoint
    )
    fsm_mult = (
        cfg.fsm_confirmation_multiplier
        if (ev.phase10_alerts_count > 0 or ev.phase10_temporal_state in _CONFIRMED_STATES)
        else 1.0
    )
    weight = sample_maturity * crop_q * stability_q * gamma * fsm_mult

    return ViewPosteriorSummary(
        camera_id=ev.camera_id,
        posterior_no_helmet=p_nh,
        log_odds_no_helmet=log_odds,
        quality_weight=weight,
        sample_maturity=sample_maturity,
        scale_quality=crop_q,
        stability_quality=stability_q,
        observations=n_obs,
    )


def _window_for_views(
    helmet_state: str,
    views: list[ViewRiderEvidence],
) -> tuple[float, float] | None:
    if helmet_state != "NO_HELMET":
        return None
    starts: list[float] = []
    ends: list[float] = []
    for ev in views:
        if ev.no_helmet_count <= 0 and ev.track_helmet_state != "NO_HELMET":
            continue
        if ev.t_start_ref_sec is not None and ev.t_end_ref_sec is not None:
            starts.append(float(ev.t_start_ref_sec))
            ends.append(float(ev.t_end_ref_sec))
    if starts and ends:
        return (min(starts), max(ends))
    return None


def _collect_meta(views: list[ViewRiderEvidence]) -> tuple[tuple[int, ...], tuple[str, ...], tuple[str, ...]]:
    raw_slots = tuple(sorted({int(v.slot_index) for v in views if v.slot_index is not None}))
    roles = tuple(sorted({v.effective_role for v in views}))
    alerts = tuple(sorted({aid for v in views for aid in v.alert_ids}))
    return raw_slots, roles, alerts


def evaluate_single_view_belief(
    ev: ViewRiderEvidence,
    *,
    canonical_slot: int,
    role: str | None = None,
    sync_reliability: float = 1.0,
    quarantined_reason: str | None = None,
    config: FusionConfig | None = None,
) -> RiderFusionResult:
    """Evaluate a single-camera rider (or quarantined view) under the unified decision gate."""
    cfg = config or FusionConfig()
    summary = compute_view_posterior_and_weight(ev, cfg)
    eff_role = role or ev.effective_role
    reasons: list[str] = []
    quarantined = quarantined_reason is not None

    if quarantined_reason:
        reasons.append(quarantined_reason)

    n_obs = ev.total_observations
    p_nh = summary.posterior_no_helmet
    if n_obs < cfg.min_observations_for_decision or n_obs <= 0:
        state_str = "UNCERTAIN"
        prov = "UNCERTAIN"
        decision = (
            FusionDecisionState.AMBIGUOUS_ASSOC_ABSTAIN
            if quarantined
            else FusionDecisionState.LOW_CONFIDENCE_ABSTAIN
        )
        abstained = True
        reasons.append(f"INSUFFICIENT_OBSERVATIONS_{n_obs}_BELOW_{cfg.min_observations_for_decision}")
    elif p_nh >= cfg.violation_prob_threshold:
        state_str = "NO_HELMET"
        prov = "AI_DIRECT"
        decision = (
            FusionDecisionState.SINGLE_VIEW_VIOLATION_FRONT
            if ev.camera_id == "front"
            else FusionDecisionState.SINGLE_VIEW_VIOLATION_BACK
        )
        abstained = False
        reasons.append(f"SINGLE_VIEW_{ev.camera_id.upper()}_VIOLATION_P_{p_nh:.3f}")
    elif p_nh <= cfg.compliant_prob_threshold:
        state_str = "HELMET"
        prov = "AI_DIRECT"
        decision = (
            FusionDecisionState.SINGLE_VIEW_COMPLIANT_FRONT
            if ev.camera_id == "front"
            else FusionDecisionState.SINGLE_VIEW_COMPLIANT_BACK
        )
        abstained = False
        reasons.append(f"SINGLE_VIEW_{ev.camera_id.upper()}_COMPLIANT_P_{p_nh:.3f}")
    else:
        state_str = "UNCERTAIN"
        prov = "UNCERTAIN"
        decision = (
            FusionDecisionState.AMBIGUOUS_ASSOC_ABSTAIN
            if quarantined
            else FusionDecisionState.LOW_CONFIDENCE_ABSTAIN
        )
        abstained = True
        reasons.append(f"LOW_CONFIDENCE_POSTERIOR_{p_nh:.3f}")

    raw_slots, rec_roles, alert_ids = _collect_meta([ev])
    return RiderFusionResult(
        canonical_slot=canonical_slot,
        role=eff_role,
        helmet_state=state_str,
        provenance=prov,
        fusion_decision_state=decision,
        fused_p_no_helmet=p_nh,
        fused_log_odds=summary.log_odds_no_helmet,
        contradiction_k=0.0,
        sync_reliability=sync_reliability,
        association_score=None,
        cross_view_fused=False,
        quarantined=quarantined,
        abstained=abstained,
        decision_reasons=tuple(reasons),
        violation_window_ref_sec=_window_for_views(state_str, [ev]),
        front_evidence=ev if ev.camera_id == "front" else None,
        back_evidence=ev if ev.camera_id == "back" else None,
        front_summary=summary if ev.camera_id == "front" else None,
        back_summary=summary if ev.camera_id == "back" else None,
        raw_subtrack_slots=raw_slots,
        recorded_slot_roles=rec_roles,
        alert_ids=alert_ids,
    )


def fuse_rider_beliefs(
    front: ViewRiderEvidence | None,
    back: ViewRiderEvidence | None,
    *,
    canonical_slot: int,
    role: str | None = None,
    association_score: float = 1.0,
    sync_reliability: float = 1.0,
    quarantined_reason: str | None = None,
    config: FusionConfig | None = None,
) -> RiderFusionResult:
    """Quality-weighted log-odds fusion for a cross-camera rider or orphan pair."""
    cfg = config or FusionConfig()
    if front is None and back is None:
        return RiderFusionResult(
            canonical_slot=canonical_slot,
            role=role or "uncertain",
            helmet_state="UNCERTAIN",
            provenance="UNCERTAIN",
            fusion_decision_state=FusionDecisionState.LOW_CONFIDENCE_ABSTAIN,
            fused_p_no_helmet=0.5,
            fused_log_odds=0.0,
            contradiction_k=0.0,
            sync_reliability=sync_reliability,
            association_score=association_score,
            cross_view_fused=False,
            quarantined=quarantined_reason is not None,
            abstained=True,
            decision_reasons=("NO_VIEW_EVIDENCE",),
        )
    if front is not None and back is None:
        return evaluate_single_view_belief(
            front,
            canonical_slot=canonical_slot,
            role=role,
            sync_reliability=sync_reliability,
            quarantined_reason=quarantined_reason,
            config=cfg,
        )
    if back is not None and front is None:
        return evaluate_single_view_belief(
            back,
            canonical_slot=canonical_slot,
            role=role,
            sync_reliability=sync_reliability,
            quarantined_reason=quarantined_reason,
            config=cfg,
        )

    assert front is not None and back is not None
    eff_role = role or (
        front.effective_role if front.effective_role != "uncertain" else back.effective_role
    )
    f_sum = compute_view_posterior_and_weight(front, cfg)
    b_sum = compute_view_posterior_and_weight(back, cfg)
    raw_slots, rec_roles, alert_ids = _collect_meta([front, back])

    # If one view has zero genuine observations, fall back cleanly to the populated view.
    if f_sum.observations == 0 and b_sum.observations == 0:
        return RiderFusionResult(
            canonical_slot=canonical_slot,
            role=eff_role,
            helmet_state="UNCERTAIN",
            provenance="UNCERTAIN",
            fusion_decision_state=FusionDecisionState.LOW_CONFIDENCE_ABSTAIN,
            fused_p_no_helmet=0.5,
            fused_log_odds=0.0,
            contradiction_k=0.0,
            sync_reliability=sync_reliability,
            association_score=association_score,
            cross_view_fused=False,
            quarantined=quarantined_reason is not None,
            abstained=True,
            decision_reasons=("ZERO_OBSERVATIONS_BOTH_VIEWS",),
            front_evidence=front,
            back_evidence=back,
            front_summary=f_sum,
            back_summary=b_sum,
            raw_subtrack_slots=raw_slots,
            recorded_slot_roles=rec_roles,
            alert_ids=alert_ids,
        )
    if f_sum.observations == 0:
        res = evaluate_single_view_belief(
            back,
            canonical_slot=canonical_slot,
            role=eff_role,
            sync_reliability=sync_reliability,
            quarantined_reason=quarantined_reason,
            config=cfg,
        )
        return RiderFusionResult(**{**res.__dict__, "front_evidence": front, "front_summary": f_sum, "raw_subtrack_slots": raw_slots, "recorded_slot_roles": rec_roles, "alert_ids": alert_ids})
    if b_sum.observations == 0:
        res = evaluate_single_view_belief(
            front,
            canonical_slot=canonical_slot,
            role=eff_role,
            sync_reliability=sync_reliability,
            quarantined_reason=quarantined_reason,
            config=cfg,
        )
        return RiderFusionResult(**{**res.__dict__, "back_evidence": back, "back_summary": b_sum, "raw_subtrack_slots": raw_slots, "recorded_slot_roles": rec_roles, "alert_ids": alert_ids})

    if quarantined_reason is not None:
        return RiderFusionResult(
            canonical_slot=canonical_slot,
            role=eff_role,
            helmet_state="UNCERTAIN",
            provenance="UNCERTAIN",
            fusion_decision_state=FusionDecisionState.AMBIGUOUS_ASSOC_ABSTAIN,
            fused_p_no_helmet=0.5,
            fused_log_odds=0.0,
            contradiction_k=compute_contradiction_metric(f_sum.posterior_no_helmet, b_sum.posterior_no_helmet),
            sync_reliability=sync_reliability,
            association_score=association_score,
            cross_view_fused=False,
            quarantined=True,
            abstained=True,
            decision_reasons=(quarantined_reason, "CROSS_VIEW_FUSION_QUARANTINED"),
            violation_window_ref_sec=None,
            front_evidence=front,
            back_evidence=back,
            front_summary=f_sum,
            back_summary=b_sum,
            raw_subtrack_slots=raw_slots,
            recorded_slot_roles=rec_roles,
            alert_ids=alert_ids,
        )

    c_assoc = max(0.05, min(1.0, float(association_score))) * max(
        cfg.min_sync_reliability, min(1.0, float(sync_reliability))
    )
    w_f = f_sum.quality_weight
    w_b = c_assoc * b_sum.quality_weight
    denom = w_f + w_b
    l_base = ((w_f * f_sum.log_odds_no_helmet) + (w_b * b_sum.log_odds_no_helmet)) / denom if denom > 1e-12 else 0.0

    corrob = 0.0
    if f_sum.log_odds_no_helmet * b_sum.log_odds_no_helmet > 0.0:
        sign = 1.0 if l_base >= 0.0 else -1.0
        corrob = (
            sign
            * cfg.corroboration_beta
            * c_assoc
            * min(abs(f_sum.log_odds_no_helmet), abs(b_sum.log_odds_no_helmet))
        )
    l_fused = l_base + corrob
    p_fused = _clamp_prob(_sigmoid(l_fused), cfg.prob_clamp_eps)
    k_conflict = compute_contradiction_metric(f_sum.posterior_no_helmet, b_sum.posterior_no_helmet)

    reasons: list[str] = []
    if k_conflict >= cfg.conflict_threshold:
        state_str = "UNCERTAIN"
        prov = "UNCERTAIN"
        decision = FusionDecisionState.EVIDENCE_CONFLICT_ABSTAIN
        abstained = True
        reasons.append(f"EVIDENCE_CONFLICT_K_{k_conflict:.3f}_GE_{cfg.conflict_threshold:.2f}")
    elif max(front.total_observations, back.total_observations) < cfg.min_observations_for_decision and (
        front.total_observations + back.total_observations
    ) < cfg.min_observations_for_decision:
        state_str = "UNCERTAIN"
        prov = "UNCERTAIN"
        decision = FusionDecisionState.LOW_CONFIDENCE_ABSTAIN
        abstained = True
        reasons.append("DUAL_INSUFFICIENT_OBSERVATIONS")
    elif p_fused >= cfg.violation_prob_threshold:
        state_str = "NO_HELMET"
        prov = "AI_DIRECT"
        decision = FusionDecisionState.VERIFIED_DUAL_VIOLATION
        abstained = False
        reasons.append(f"DUAL_FUSED_VIOLATION_P_{p_fused:.3f}")
    elif p_fused <= cfg.compliant_prob_threshold:
        state_str = "HELMET"
        prov = "AI_DIRECT"
        decision = FusionDecisionState.VERIFIED_DUAL_COMPLIANT
        abstained = False
        reasons.append(f"DUAL_FUSED_COMPLIANT_P_{p_fused:.3f}")
    else:
        state_str = "UNCERTAIN"
        prov = "UNCERTAIN"
        decision = FusionDecisionState.LOW_CONFIDENCE_ABSTAIN
        abstained = True
        reasons.append(f"DUAL_LOW_CONFIDENCE_P_{p_fused:.3f}")

    return RiderFusionResult(
        canonical_slot=canonical_slot,
        role=eff_role,
        helmet_state=state_str,
        provenance=prov,
        fusion_decision_state=decision,
        fused_p_no_helmet=p_fused,
        fused_log_odds=l_fused,
        contradiction_k=k_conflict,
        sync_reliability=sync_reliability,
        association_score=association_score,
        cross_view_fused=True,
        quarantined=False,
        abstained=abstained,
        decision_reasons=tuple(reasons),
        violation_window_ref_sec=_window_for_views(state_str, [front, back]),
        front_evidence=front,
        back_evidence=back,
        front_summary=f_sum,
        back_summary=b_sum,
        raw_subtrack_slots=raw_slots,
        recorded_slot_roles=rec_roles,
        alert_ids=alert_ids,
    )


__all__ = [
    "FusionConfig",
    "FusionDecisionState",
    "RiderFusionResult",
    "ViewPosteriorSummary",
    "ViewRiderEvidence",
    "compute_contradiction_metric",
    "compute_mean_head_area_from_history",
    "compute_sync_reliability",
    "compute_view_posterior_and_weight",
    "evaluate_single_view_belief",
    "fuse_rider_beliefs",
]
