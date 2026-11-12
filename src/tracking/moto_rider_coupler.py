"""Spatial Rider-to-Motorcycle Coupler (`MotoRiderCoupler`).

Links cranial detections (`helmet` / `no_helmet` from Phase 9 `best.pt`) and full-body
`person` detections (from COCO `yolov8n.pt`) to parent `motorcycle` bounding boxes
within a single camera frame.

Scientific Note:
All geometric thresholds and scoring weights in `CouplerConfig` are initial
engineering parameters (hypotheses) configurable via `configs/tracking_config.yaml`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.detection.camera_detector import StructuredDetection


@dataclass
class CouplerConfig:
    """Configurable engineering parameters for spatial rider-to-motorcycle coupling."""
    max_riders_per_motorcycle: int = 3
    min_coupling_score: float = 0.35
    ambiguity_margin_threshold: float = 0.08
    head_upper_cone_height_ratio: float = 1.25
    head_horizontal_expansion_ratio: float = 0.25
    max_head_center_below_moto_ratio: float = 0.55
    max_head_to_moto_width_ratio: float = 1.15
    person_mount_min_iou: float = 0.12
    person_mount_min_ioa: float = 0.25
    head_in_person_upper_ratio: float = 0.45
    head_in_person_min_ioa: float = 0.35
    head_person_ambiguity_margin: float = 0.08
    slot_ordering_min_separation_px: float = 8.0
    provisional_single_rider_unconfident: bool = True
    # Weights when a mounted person box bridges the head and motorcycle
    w_with_person_cone: float = 0.40
    w_with_person_bridge: float = 0.30
    w_with_person_lateral: float = 0.20
    w_with_person_vertical: float = 0.10
    # Weights when person box is occluded/missing
    w_no_person_cone: float = 0.55
    w_no_person_lateral: float = 0.30
    w_no_person_vertical: float = 0.15


@dataclass(frozen=True)
class CouplingGeometricFeatures:
    """Detailed geometric features and diagnostic scores for a (head, motorcycle) pair."""
    moto_idx: int
    cone_ioa: float
    person_bridge_score: float
    matched_person_idx: int | None
    lateral_alignment_score: float
    vertical_scale_score: float
    normalized_dx: float
    normalized_dy: float
    head_to_moto_width_ratio: float
    composite_score: float
    rejection_reason: str | None
    head_person_abstained: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "moto_idx": self.moto_idx,
            "cone_ioa": round(self.cone_ioa, 4),
            "person_bridge_score": round(self.person_bridge_score, 4),
            "matched_person_idx": self.matched_person_idx,
            "head_person_abstained": self.head_person_abstained,
            "lateral_alignment_score": round(self.lateral_alignment_score, 4),
            "vertical_scale_score": round(self.vertical_scale_score, 4),
            "normalized_dx": round(self.normalized_dx, 4),
            "normalized_dy": round(self.normalized_dy, 4),
            "head_to_moto_width_ratio": round(self.head_to_moto_width_ratio, 4),
            "composite_score": round(self.composite_score, 4),
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class CoupledRiderObservation:
    """A rider head observation successfully coupled to a parent motorcycle."""
    head_detection: StructuredDetection
    person_detection: StructuredDetection | None
    slot_index: int  # 0, 1, or 2
    slot_role: str  # "driver", "pillion_1", "pillion_2", or "ambiguous_slot_<k>"
    ordering_confident: bool
    coupling_score: float
    second_best_score: float
    margin_over_second_best: float
    geometric_features: CouplingGeometricFeatures
    slot_role_provisional: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_index": self.slot_index,
            "slot_role": self.slot_role,
            "ordering_confident": self.ordering_confident,
            "coupling_score": round(self.coupling_score, 4),
            "second_best_score": round(self.second_best_score, 4),
            "margin_over_second_best": round(self.margin_over_second_best, 4),
            "head": self.head_detection.to_dict(),
            "person": self.person_detection.to_dict() if self.person_detection else None,
            "geometric_features": self.geometric_features.to_dict(),
        }


@dataclass(frozen=True)
class CoupledMotorcycleObservation:
    """A motorcycle detection paired with its 0..3 coupled rider observations."""
    moto_index: int
    motorcycle_detection: StructuredDetection
    mounted_persons: list[StructuredDetection]
    riders: list[CoupledRiderObservation]

    @property
    def rider_count(self) -> int:
        return len(self.riders)

    def to_dict(self) -> dict[str, Any]:
        return {
            "moto_index": self.moto_index,
            "motorcycle": self.motorcycle_detection.to_dict(),
            "rider_count": self.rider_count,
            "mounted_persons_count": len(self.mounted_persons),
            "riders": [r.to_dict() for r in self.riders],
        }


@dataclass(frozen=True)
class OrphanRiderObservation:
    """
    A rider head observation that could NOT be reliably coupled to a single motorcycle
    (e.g., occluded motorcycle chassis, below coupling threshold, or ambiguous between
    two adjacent motorcycles).
    """
    head_detection: StructuredDetection
    person_detection: StructuredDetection | None
    rejection_reason: str
    best_candidate_moto_idx: int | None
    best_candidate_score: float
    second_best_score: float
    candidate_features: CouplingGeometricFeatures | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "head": self.head_detection.to_dict(),
            "person": self.person_detection.to_dict() if self.person_detection else None,
            "rejection_reason": self.rejection_reason,
            "best_candidate_moto_idx": self.best_candidate_moto_idx,
            "best_candidate_score": round(self.best_candidate_score, 4),
            "second_best_score": round(self.second_best_score, 4),
            "candidate_features": (
                self.candidate_features.to_dict() if self.candidate_features else None
            ),
        }


@dataclass(frozen=True)
class FrameCouplingResult:
    """Complete output of MotoRiderCoupler for a single camera frame."""
    camera_id: str
    frame_index: int
    pts_timestamp_sec: float
    coupled_motorcycles: list[CoupledMotorcycleObservation]
    orphan_riders: list[OrphanRiderObservation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "frame_index": self.frame_index,
            "pts_timestamp_sec": round(self.pts_timestamp_sec, 6),
            "num_coupled_motorcycles": len(self.coupled_motorcycles),
            "num_coupled_riders": sum(m.rider_count for m in self.coupled_motorcycles),
            "num_orphan_riders": len(self.orphan_riders),
            "coupled_motorcycles": [m.to_dict() for m in self.coupled_motorcycles],
            "orphan_riders": [o.to_dict() for o in self.orphan_riders],
        }


class MotoRiderCoupler:
    """
    Testable spatial coupler associating rider heads and persons with parent motorcycles.
    """

    def __init__(self, config: CouplerConfig | None = None) -> None:
        self.config = config or CouplerConfig()

    def _compute_person_moto_affinity(
        self,
        person: StructuredDetection,
        moto: StructuredDetection,
    ) -> float:
        """Evaluate whether a person detection is mounted on a motorcycle."""
        if moto.width <= 1e-4 or moto.height <= 1e-4:
            return 0.0
        px, _ = person.center
        mx1_exp = moto.x1 - self.config.head_horizontal_expansion_ratio * moto.width
        mx2_exp = moto.x2 + self.config.head_horizontal_expansion_ratio * moto.width
        if px < mx1_exp or px > mx2_exp:
            return 0.0

        iou_val = person.iou(moto)
        ioa_val = person.intersection_over_self_area(moto)
        if iou_val < self.config.person_mount_min_iou and ioa_val < self.config.person_mount_min_ioa:
            return 0.0

        lat_dist = abs(px - moto.center[0]) / max(1.0, 0.5 * moto.width)
        lat_score = max(0.0, 1.0 - 0.5 * (lat_dist ** 2))
        return min(1.0, 0.45 * min(1.0, ioa_val / 0.5) + 0.30 * min(1.0, iou_val / 0.35) + 0.25 * lat_score)

    def _compute_head_in_person_score(
        self,
        head: StructuredDetection,
        person: StructuredDetection,
    ) -> float:
        """Evaluate whether a head box sits inside the upper cranial region of a person box."""
        if person.width <= 1e-4 or person.height <= 1e-4:
            return 0.0
        upper_y2 = person.y1 + self.config.head_in_person_upper_ratio * person.height
        upper_box = (
            person.x1 - 0.10 * person.width,
            person.y1 - 0.15 * person.height,
            person.x2 + 0.10 * person.width,
            upper_y2,
        )
        ioa_upper = head.intersection_over_self_area(upper_box)
        if ioa_upper < self.config.head_in_person_min_ioa:
            return 0.0
        dx = abs(head.center[0] - person.center[0]) / max(1.0, 0.5 * person.width)
        lat_factor = max(0.0, 1.0 - 0.5 * (dx ** 2))
        dy_apex = abs(head.y1 - person.y1) / max(1.0, head.height)
        apex_factor = max(0.0, 1.0 - dy_apex)
        return float(min(1.0, 0.50 * ioa_upper + 0.25 * lat_factor + 0.25 * apex_factor))

    def assign_heads_to_persons_bipartite(
        self,
        heads: list[StructuredDetection],
        persons: list[StructuredDetection],
    ) -> tuple[dict[int, tuple[int, float]], set[int]]:
        """
        Perform gated one-to-one Hungarian bipartite assignment between detected
        rider heads and detected person boxes within a single frame.

        Enforces:
        1. Strict one-to-one matching: one person cannot be assigned to multiple heads,
           and one head cannot be assigned to multiple persons.
        2. Minimum cranial inclusion gate (`head_in_person_min_ioa`).
        3. Competitive ambiguity abstention (`head_person_ambiguity_margin`): if an
           unmatched head competes for the same person within the margin, or a head
           competes between two unmatched persons within the margin, or a 2x2 swap
           has margin below threshold, the person bridge abstains rather than forcing
           an arbitrary head-to-person link.

        Returns:
            (head_to_person_map, abstained_head_indices)
            where head_to_person_map maps head_idx -> (person_idx, hp_score).
        """
        n_h = len(heads)
        n_p = len(persons)
        if n_h == 0 or n_p == 0:
            return {}, set()

        score_mat = np.zeros((n_h, n_p), dtype=np.float64)
        for h_i, h_det in enumerate(heads):
            for p_j, p_det in enumerate(persons):
                score_mat[h_i, p_j] = self._compute_head_in_person_score(h_det, p_det)

        cost_mat = np.full((n_h, n_p), 1e6, dtype=np.float64)
        for h_i in range(n_h):
            for p_j in range(n_p):
                if score_mat[h_i, p_j] > 0.0:
                    cost_mat[h_i, p_j] = 1.0 - score_mat[h_i, p_j]

        row_ind, col_ind = linear_sum_assignment(cost_mat)
        tentative_pairs: list[tuple[int, int, float]] = []
        matched_h: set[int] = set()
        matched_p: set[int] = set()
        for r, c in zip(row_ind, col_ind):
            if cost_mat[r, c] < 1.0:
                tentative_pairs.append((int(r), int(c), float(score_mat[r, c])))
                matched_h.add(int(r))
                matched_p.add(int(c))

        abstained_heads: set[int] = set()
        margin_thresh = self.config.head_person_ambiguity_margin

        # Check competitive ambiguity on each tentative match
        rejected_pairs: set[tuple[int, int]] = set()
        for h_i, p_j, sc in tentative_pairs:
            # 1. Does an unmatched head (or another head whose own match isn't stronger)
            #    compete for this same person p_j within margin_thresh?
            for h_other in range(n_h):
                if h_other == h_i:
                    continue
                other_sc = float(score_mat[h_other, p_j])
                if other_sc <= 0.0:
                    continue
                # If h_other is unmatched, or if h_other has a nearly identical score on p_j
                if h_other not in matched_h and (sc - other_sc) < margin_thresh:
                    rejected_pairs.add((h_i, p_j))
                    abstained_heads.add(h_i)
                    abstained_heads.add(h_other)

            # 2. Does an unmatched person p_other compete for this same head h_i within margin_thresh?
            for p_other in range(n_p):
                if p_other == p_j:
                    continue
                other_sc = float(score_mat[h_i, p_other])
                if other_sc <= 0.0:
                    continue
                if p_other not in matched_p and (sc - other_sc) < margin_thresh:
                    rejected_pairs.add((h_i, p_j))
                    abstained_heads.add(h_i)

        # 3. Check 2x2 swap ambiguity between pairs of tentative matches
        for idx_a in range(len(tentative_pairs)):
            h_a, p_a, sc_aa = tentative_pairs[idx_a]
            for idx_b in range(idx_a + 1, len(tentative_pairs)):
                h_b, p_b, sc_bb = tentative_pairs[idx_b]
                sc_ab = float(score_mat[h_a, p_b])
                sc_ba = float(score_mat[h_b, p_a])
                if sc_ab > 0.0 and sc_ba > 0.0:
                    gain = (sc_aa + sc_bb) - (sc_ab + sc_ba)
                    if gain < margin_thresh:
                        rejected_pairs.add((h_a, p_a))
                        rejected_pairs.add((h_b, p_b))
                        abstained_heads.add(h_a)
                        abstained_heads.add(h_b)

        head_to_person_map: dict[int, tuple[int, float]] = {}
        for h_i, p_j, sc in tentative_pairs:
            if (h_i, p_j) not in rejected_pairs and h_i not in abstained_heads:
                head_to_person_map[h_i] = (p_j, sc)

        return head_to_person_map, abstained_heads

    def evaluate_head_moto_pair(
        self,
        head: StructuredDetection,
        moto: StructuredDetection,
        moto_idx: int,
        persons: list[StructuredDetection],
        person_moto_scores: dict[tuple[int, int], float],
        assigned_person_for_head: tuple[int, float] | None = None,
        head_person_abstained: bool = False,
        use_exclusive_person_assignment: bool = False,
    ) -> CouplingGeometricFeatures:
        """Compute geometric coupling features and composite score between head and moto."""
        mw = max(1.0, moto.width)
        mh = max(1.0, moto.height)
        hx, hy = head.center
        mx, my = moto.center

        half_span_x = (0.5 + self.config.head_horizontal_expansion_ratio) * mw
        norm_dx = abs(hx - mx) / half_span_x

        # Upper rider envelope cone above and overlapping the upper saddle of the motorcycle
        cone_x1 = moto.x1 - self.config.head_horizontal_expansion_ratio * mw
        cone_x2 = moto.x2 + self.config.head_horizontal_expansion_ratio * mw
        cone_y1 = moto.y1 - self.config.head_upper_cone_height_ratio * mh
        cone_y2 = moto.y1 + self.config.max_head_center_below_moto_ratio * mh
        cone_box = (cone_x1, cone_y1, cone_x2, cone_y2)

        cone_ioa = head.intersection_over_self_area(cone_box)
        width_ratio = head.width / mw
        norm_dy = (moto.y1 - hy) / mh  # Positive when head center is above moto top edge

        # Hard geometric gates
        if norm_dx > 1.15:
            return CouplingGeometricFeatures(
                moto_idx=moto_idx,
                cone_ioa=cone_ioa,
                person_bridge_score=0.0,
                matched_person_idx=None,
                lateral_alignment_score=0.0,
                vertical_scale_score=0.0,
                normalized_dx=norm_dx,
                normalized_dy=norm_dy,
                head_to_moto_width_ratio=width_ratio,
                composite_score=0.0,
                rejection_reason="LATERAL_OUT_OF_BOUNDS",
                head_person_abstained=head_person_abstained,
            )

        if hy < cone_y1 or hy > cone_y2:
            return CouplingGeometricFeatures(
                moto_idx=moto_idx,
                cone_ioa=cone_ioa,
                person_bridge_score=0.0,
                matched_person_idx=None,
                lateral_alignment_score=0.0,
                vertical_scale_score=0.0,
                normalized_dx=norm_dx,
                normalized_dy=norm_dy,
                head_to_moto_width_ratio=width_ratio,
                composite_score=0.0,
                rejection_reason="VERTICAL_OUT_OF_BOUNDS",
                head_person_abstained=head_person_abstained,
            )

        if width_ratio > self.config.max_head_to_moto_width_ratio:
            return CouplingGeometricFeatures(
                moto_idx=moto_idx,
                cone_ioa=cone_ioa,
                person_bridge_score=0.0,
                matched_person_idx=None,
                lateral_alignment_score=0.0,
                vertical_scale_score=0.0,
                normalized_dx=norm_dx,
                normalized_dy=norm_dy,
                head_to_moto_width_ratio=width_ratio,
                composite_score=0.0,
                rejection_reason="HEAD_SCALE_IMPLAUSIBLE",
                head_person_abstained=head_person_abstained,
            )

        lateral_score = max(0.0, 1.0 - (norm_dx ** 2))

        # Ideal head vertical position is near or slightly above moto.y1 (norm_dy in [-0.35, +0.85])
        if -0.35 <= norm_dy <= 0.85:
            vert_pos_score = 1.0
        else:
            dist_from_band = min(abs(norm_dy - (-0.35)), abs(norm_dy - 0.85))
            vert_pos_score = max(0.0, 1.0 - dist_from_band / 0.60)

        scale_plausibility = 1.0 if 0.12 <= width_ratio <= 0.85 else 0.65
        vertical_scale_score = vert_pos_score * scale_plausibility

        # Evaluate person bridge using the 1-to-1 Hungarian bipartite head-to-person assignment
        best_p_score = 0.0
        best_p_idx: int | None = None
        if use_exclusive_person_assignment:
            if assigned_person_for_head is not None and not head_person_abstained:
                p_idx, hp_score = assigned_person_for_head
                pm_score = person_moto_scores.get((p_idx, moto_idx), 0.0)
                if pm_score > 0.0 and hp_score > 0.0:
                    best_p_score = float((pm_score * hp_score) ** 0.5)
                    best_p_idx = p_idx
        else:
            # Fallback if evaluate_head_moto_pair is called standalone outside couple_frame
            for p_idx, p_det in enumerate(persons):
                pm_score = person_moto_scores.get((p_idx, moto_idx), 0.0)
                if pm_score <= 0.0:
                    continue
                hp_score = self._compute_head_in_person_score(head, p_det)
                if hp_score > 0.0:
                    bridge = float((pm_score * hp_score) ** 0.5)
                    if bridge > best_p_score:
                        best_p_score = bridge
                        best_p_idx = p_idx

        if best_p_score > 0.0:
            comp = (
                self.config.w_with_person_cone * cone_ioa
                + self.config.w_with_person_bridge * best_p_score
                + self.config.w_with_person_lateral * lateral_score
                + self.config.w_with_person_vertical * vertical_scale_score
            )
        else:
            comp = (
                self.config.w_no_person_cone * cone_ioa
                + self.config.w_no_person_lateral * lateral_score
                + self.config.w_no_person_vertical * vertical_scale_score
            )

        rej = None if comp >= self.config.min_coupling_score else "BELOW_MIN_COUPLING_SCORE"
        return CouplingGeometricFeatures(
            moto_idx=moto_idx,
            cone_ioa=cone_ioa,
            person_bridge_score=best_p_score,
            matched_person_idx=best_p_idx,
            lateral_alignment_score=lateral_score,
            vertical_scale_score=vertical_scale_score,
            normalized_dx=norm_dx,
            normalized_dy=norm_dy,
            head_to_moto_width_ratio=width_ratio,
            composite_score=float(comp),
            rejection_reason=rej,
            head_person_abstained=head_person_abstained,
        )

    def _order_rider_slots(
        self,
        camera_id: str,
        raw_riders: list[tuple[StructuredDetection, StructuredDetection | None, float, float, CouplingGeometricFeatures]],
    ) -> list[CoupledRiderObservation]:
        """
        Assign slot_index (0, 1, 2) and slot_role ("driver", "pillion_1", "pillion_2")
        only when geometric separation supports confident ordering.
        """
        if not raw_riders:
            return []

        if len(raw_riders) == 1:
            h, p, sc, s2, feat = raw_riders[0]
            return [
                CoupledRiderObservation(
                    head_detection=h,
                    person_detection=p,
                    slot_index=0,
                    slot_role="driver",
                    ordering_confident=not self.config.provisional_single_rider_unconfident,
                    coupling_score=sc,
                    second_best_score=s2,
                    margin_over_second_best=sc - s2,
                    geometric_features=feat,
                    slot_role_provisional=True,
                )
            ]

        # In oncoming front view, driver is closer/lower in frame (larger center_y) or larger head height;
        # in receding back view, driver is further ahead/higher in frame (smaller center_y).
        if camera_id == "front":
            sorted_riders = sorted(
                raw_riders,
                key=lambda item: (item[0].center[1], item[0].height),
                reverse=True,
            )
        else:
            sorted_riders = sorted(
                raw_riders,
                key=lambda item: (item[0].center[1], -item[0].height),
                reverse=False,
            )

        # Check whether vertical or horizontal separation between adjacent slots is >= min_separation_px
        confident = True
        for idx in range(len(sorted_riders) - 1):
            h_a = sorted_riders[idx][0]
            h_b = sorted_riders[idx + 1][0]
            dy = abs(h_a.center[1] - h_b.center[1])
            dh = abs(h_a.height - h_b.height)
            if dy < self.config.slot_ordering_min_separation_px and dh < self.config.slot_ordering_min_separation_px:
                confident = False
                break

        role_names = ["driver", "pillion_1", "pillion_2"]
        ordered_obs: list[CoupledRiderObservation] = []
        for s_idx, (h, p, sc, s2, feat) in enumerate(sorted_riders):
            role = role_names[s_idx] if (confident and s_idx < len(role_names)) else f"ambiguous_slot_{s_idx}"
            ordered_obs.append(
                CoupledRiderObservation(
                    head_detection=h,
                    person_detection=p,
                    slot_index=s_idx,
                    slot_role=role,
                    ordering_confident=confident,
                    coupling_score=sc,
                    second_best_score=s2,
                    margin_over_second_best=sc - s2,
                    geometric_features=feat,
                )
            )
        return ordered_obs

    def couple_frame(
        self,
        camera_id: str,
        frame_index: int,
        pts_timestamp_sec: float,
        motorcycles: list[StructuredDetection],
        persons: list[StructuredDetection],
        heads: list[StructuredDetection],
    ) -> FrameCouplingResult:
        """
        Couple detected rider heads and persons to parent motorcycles in a single frame.
        """
        # Gated one-to-one Hungarian bipartite head-to-person assignment
        head_to_person_map, abstained_heads = self.assign_heads_to_persons_bipartite(heads, persons)

        # Case 0: No motorcycles detected -> all heads become explicit OrphanRiderObservations
        if not motorcycles:
            orphans: list[OrphanRiderObservation] = []
            for h_idx, h in enumerate(heads):
                assigned_p = head_to_person_map.get(h_idx)
                best_p = persons[assigned_p[0]] if assigned_p is not None else None
                orphans.append(
                    OrphanRiderObservation(
                        head_detection=h,
                        person_detection=best_p,
                        rejection_reason="NO_MOTORCYCLE_IN_FRAME",
                        best_candidate_moto_idx=None,
                        best_candidate_score=0.0,
                        second_best_score=0.0,
                        candidate_features=None,
                    )
                )
            return FrameCouplingResult(
                camera_id=camera_id,
                frame_index=frame_index,
                pts_timestamp_sec=pts_timestamp_sec,
                coupled_motorcycles=[],
                orphan_riders=orphans,
            )

        # Precompute person-to-motorcycle mounting affinities (exclusive best moto per person)
        person_moto_scores: dict[tuple[int, int], float] = {}
        mounted_persons_per_moto: dict[int, list[StructuredDetection]] = {
            m_idx: [] for m_idx in range(len(motorcycles))
        }
        for p_idx, p_det in enumerate(persons):
            best_m = -1
            best_pm = 0.0
            for m_idx, m_det in enumerate(motorcycles):
                sc = self._compute_person_moto_affinity(p_det, m_det)
                person_moto_scores[(p_idx, m_idx)] = sc
                if sc > best_pm:
                    best_pm = sc
                    best_m = m_idx
            if best_m >= 0 and best_pm > 0.0:
                mounted_persons_per_moto[best_m].append(p_det)

        # Evaluate each head against all motorcycles using 1-to-1 head-to-person map
        tentative_per_moto: dict[
            int,
            list[tuple[StructuredDetection, StructuredDetection | None, float, float, CouplingGeometricFeatures]],
        ] = {m_idx: [] for m_idx in range(len(motorcycles))}
        orphans = []

        for h_idx, h in enumerate(heads):
            assigned_p = head_to_person_map.get(h_idx)
            hp_abstained = h_idx in abstained_heads
            feats_by_moto: list[CouplingGeometricFeatures] = [
                self.evaluate_head_moto_pair(
                    h,
                    m_det,
                    m_idx,
                    persons,
                    person_moto_scores,
                    assigned_person_for_head=assigned_p,
                    head_person_abstained=hp_abstained,
                    use_exclusive_person_assignment=True,
                )
                for m_idx, m_det in enumerate(motorcycles)
            ]
            feats_sorted = sorted(feats_by_moto, key=lambda f: f.composite_score, reverse=True)
            best_f = feats_sorted[0]
            second_f = feats_sorted[1] if len(feats_sorted) > 1 else None
            second_score = second_f.composite_score if second_f is not None else 0.0

            if best_f.matched_person_idx is not None:
                matched_person = persons[best_f.matched_person_idx]
            elif assigned_p is not None and not hp_abstained:
                matched_person = persons[assigned_p[0]]
            else:
                matched_person = None

            # Gate 1: Minimum coupling score check
            if best_f.composite_score < self.config.min_coupling_score:
                reason = best_f.rejection_reason or "BELOW_MIN_COUPLING_SCORE"
                orphans.append(
                    OrphanRiderObservation(
                        head_detection=h,
                        person_detection=matched_person,
                        rejection_reason=reason,
                        best_candidate_moto_idx=best_f.moto_idx,
                        best_candidate_score=best_f.composite_score,
                        second_best_score=second_score,
                        candidate_features=best_f,
                    )
                )
                continue

            # Gate 2: Adjacent motorcycle ambiguity abstention check
            if (
                second_f is not None
                and second_score >= self.config.min_coupling_score
                and (best_f.composite_score - second_score) < self.config.ambiguity_margin_threshold
            ):
                orphans.append(
                    OrphanRiderObservation(
                        head_detection=h,
                        person_detection=matched_person,
                        rejection_reason="AMBIGUOUS_ADJACENT_MOTORCYCLES",
                        best_candidate_moto_idx=best_f.moto_idx,
                        best_candidate_score=best_f.composite_score,
                        second_best_score=second_score,
                        candidate_features=best_f,
                    )
                )
                continue

            tentative_per_moto[best_f.moto_idx].append(
                (h, matched_person, best_f.composite_score, second_score, best_f)
            )

        # Gate 3: Enforce max_riders_per_motorcycle capacity (1..3 slots) and order slots
        coupled_motos: list[CoupledMotorcycleObservation] = []
        for m_idx, m_det in enumerate(motorcycles):
            candidates = tentative_per_moto[m_idx]
            if len(candidates) > self.config.max_riders_per_motorcycle:
                # Keep top max_riders_per_motorcycle by coupling score; demote excess to orphans
                candidates_by_score = sorted(candidates, key=lambda x: x[2], reverse=True)
                kept = candidates_by_score[: self.config.max_riders_per_motorcycle]
                excess = candidates_by_score[self.config.max_riders_per_motorcycle :]
                for h_ex, p_ex, sc_ex, s2_ex, feat_ex in excess:
                    orphans.append(
                        OrphanRiderObservation(
                            head_detection=h_ex,
                            person_detection=p_ex,
                            rejection_reason="EXCEEDED_MAX_RIDERS_PER_MOTO",
                            best_candidate_moto_idx=m_idx,
                            best_candidate_score=sc_ex,
                            second_best_score=s2_ex,
                            candidate_features=feat_ex,
                        )
                    )
                candidates = kept

            ordered_riders = self._order_rider_slots(camera_id, candidates)
            coupled_motos.append(
                CoupledMotorcycleObservation(
                    moto_index=m_idx,
                    motorcycle_detection=m_det,
                    mounted_persons=mounted_persons_per_moto[m_idx],
                    riders=ordered_riders,
                )
            )

        return FrameCouplingResult(
            camera_id=camera_id,
            frame_index=frame_index,
            pts_timestamp_sec=pts_timestamp_sec,
            coupled_motorcycles=coupled_motos,
            orphan_riders=orphans,
        )
