"""Optional Viewpoint-Robust Appearance Descriptor Extraction and Similarity (Cue C).

Extracts azimuth-invariant HSV + achromatic luminance histograms strictly from
observed (non-predicted) frames passing minimum resolution and Laplacian sharpness
quality gates:
1. Cranial Cap Core (inner 60% of observed head bounding box).
2. Upper Torso Zone (upper-middle torso region of coupled person box or inferred torso zone).
3. Motorcycle Chassis Core (lower-central 50% of motorcycle bounding box).

Strict Invariants:
- Never uses helmet classification labels (`helmet` vs. `no_helmet`) as an identity feature.
- Never extracts appearance from Kalman-predicted (`LOST`) frames.
- When frame crops are not provided or fail quality thresholds, returns
  `available = False` with an explicit `unavailable_reason` so Cue C is cleanly
  excluded and renormalized rather than producing a silent zero penalty.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import cv2
import numpy as np

from src.association.coordinate_space import (
    CropResult,
    REJ_UNKNOWN_DIMS_MISMATCH,
    UNKNOWN_DIMS_TOLERANCE_PX,
    crop_native_box,
    normalize_dimensions,
)
from src.association.tracklet_models import AppearanceDescriptorRecord, ObservedFrameRecord


@dataclass
class AppearanceConfig:
    """Configuration for optional viewpoint-robust appearance descriptors."""
    enabled: bool = True
    min_crop_width_px: int = 20
    min_crop_height_px: int = 20
    min_laplacian_sharpness: float = 12.0
    min_valid_crop_quality: float = 0.35
    hsv_h_bins: int = 12
    hsv_s_bins: int = 4
    achromatic_v_bins: int = 8
    low_saturation_threshold: int = 45
    weight_cranial_cap: float = 0.30
    weight_upper_torso: float = 0.45
    weight_moto_chassis: float = 0.25


def _extract_subbox_crop(
    image_bgr: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    x_rel_range: tuple[float, float] = (0.0, 1.0),
    y_rel_range: tuple[float, float] = (0.0, 1.0),
    native_wh: tuple[int, int] | None = None,
) -> np.ndarray | None:
    """Slice a relative sub-region of a native-coordinate `bbox_xyxy` from `image_bgr`.

    `native_wh` is the `(width, height)` of the frame the box coordinates live in; the
    stored image may be a downscaled copy of it. With `native_wh=None` the image is
    treated as native only when the box is consistent with it (see
    `src.association.coordinate_space`). Returns None on any rejection; use
    `crop_native_box` directly to obtain the rejection reason.
    """
    return crop_native_box(image_bgr, bbox_xyxy, native_wh, x_rel_range, y_rel_range).crop


def compute_crop_quality(
    crop_bgr: np.ndarray | None,
    config: AppearanceConfig,
    min_dim_scale: float = 1.0,
) -> tuple[bool, float, str | None]:
    """
    Evaluate whether an observed crop passes minimum resolution and sharpness checks.
    Returns `(is_valid, quality_score_in_0_1, rejection_reason)`.

    `min_dim_scale` (<= 1.0) is the native->stored-image scale of the keyframe the crop
    came from. The configured minimum crop size is defined in native pixels, so it is
    scaled by the same factor (floor 8 px) when the keyframe is downscaled
    (Argon TASK-05B §10). With the default 1.0 behaviour is unchanged.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return False, 0.0, "EMPTY_CROP"
    scale = min(1.0, float(min_dim_scale))
    if scale < 1.0:
        min_w = max(8, int(round(config.min_crop_width_px * scale)))
        min_h = max(8, int(round(config.min_crop_height_px * scale)))
    else:
        min_w, min_h = config.min_crop_width_px, config.min_crop_height_px
    ch, cw = crop_bgr.shape[:2]
    if cw < min_w or ch < min_h:
        return False, 0.0, "CROP_BELOW_MIN_RESOLUTION"

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < config.min_laplacian_sharpness:
        return False, 0.0, "CROP_MOTION_BLUR_BELOW_SHARPNESS"

    size_factor = min(1.0, float(cw * ch) / float(max(1, min_w * min_h * 4)))
    sharp_factor = min(1.0, lap_var / max(1.0, config.min_laplacian_sharpness * 4.0))
    quality = float(0.55 * size_factor + 0.45 * sharp_factor)
    if quality < config.min_valid_crop_quality:
        return False, quality, "CROP_COMPOSITE_QUALITY_LOW"
    return True, quality, None


def compute_hsv_achromatic_histogram(
    crop_bgr: np.ndarray,
    config: AppearanceConfig,
) -> np.ndarray:
    """
    Compute an L1-normalized joint chromatic (H x S) + achromatic (V) histogram
    vector of length `hsv_h_bins * hsv_s_bins + achromatic_v_bins`.
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    low_sat_mask = s_ch < config.low_saturation_threshold
    chrom_mask = ~low_sat_mask

    n_hs = config.hsv_h_bins * config.hsv_s_bins
    n_v = config.achromatic_v_bins
    hist_vec = np.zeros(n_hs + n_v, dtype=np.float64)

    total_px = float(max(1, crop_bgr.shape[0] * crop_bgr.shape[1]))

    if np.any(chrom_mask):
        h_vals = h_ch[chrom_mask].astype(np.float64)
        s_vals = s_ch[chrom_mask].astype(np.float64)
        hist_hs, _, _ = np.histogram2d(
            h_vals,
            s_vals,
            bins=[config.hsv_h_bins, config.hsv_s_bins],
            range=[[0.0, 180.0], [float(config.low_saturation_threshold), 256.0]],
        )
        hist_vec[:n_hs] = hist_hs.ravel() / total_px

    if np.any(low_sat_mask):
        v_vals = v_ch[low_sat_mask].astype(np.float64)
        hist_v, _ = np.histogram(
            v_vals,
            bins=config.achromatic_v_bins,
            range=(0.0, 256.0),
        )
        hist_vec[n_hs:] = hist_v / total_px

    norm = float(np.sum(hist_vec))
    if norm > 1e-9:
        hist_vec /= norm
    return hist_vec


def bhattacharyya_histogram_similarity(
    hist_a: list[float] | np.ndarray,
    hist_b: list[float] | np.ndarray,
) -> float | None:
    """
    Compute Bhattacharyya coefficient $\\sum \\sqrt{p_i q_i} \\in [0, 1]$ between two
    L1-normalized histograms. Returns `None` if either histogram is empty or invalid.
    """
    if not hist_a or not hist_b or len(hist_a) != len(hist_b):
        return None
    a = np.asarray(hist_a, dtype=np.float64)
    b = np.asarray(hist_b, dtype=np.float64)
    sa = float(np.sum(a))
    sb = float(np.sum(b))
    if sa <= 1e-9 or sb <= 1e-9:
        return None
    a = a / sa
    b = b / sb
    bc = float(np.sum(np.sqrt(np.maximum(0.0, a * b))))
    return float(np.clip(bc, 0.0, 1.0))


def extract_tracklet_appearance_descriptor(
    observed_frames: list[ObservedFrameRecord],
    frame_images_by_index: dict[int, np.ndarray] | None,
    config: AppearanceConfig,
    is_motorcycle_tracklet: bool = True,
    native_frame_dimensions: tuple[int, int] | None = None,
) -> AppearanceDescriptorRecord:
    """
    Extract quality-weighted average appearance histograms strictly from observed
    frames (`ObservedFrameRecord`) when `frame_images_by_index` is supplied.

    `native_frame_dimensions` is the `(width, height)` of the decoded frame in which the
    record bounding boxes live. Stored keyframes may be downscaled copies; box
    coordinates are mapped into each image's actual coordinate system via
    `src.association.coordinate_space`.
    """
    if not config.enabled:
        return AppearanceDescriptorRecord(
            available=False,
            unavailable_reason="CUE_DISABLED_IN_CONFIG",
        )
    if not frame_images_by_index:
        return AppearanceDescriptorRecord(
            available=False,
            unavailable_reason="NO_FRAME_IMAGES_PROVIDED",
        )
    if not observed_frames:
        return AppearanceDescriptorRecord(
            available=False,
            unavailable_reason="NO_OBSERVED_FRAMES",
        )

    native_wh = normalize_dimensions(native_frame_dimensions)

    # All boxes belonging to this tracklet so that, when native_wh is None,
    # an out-of-bounds box on *any* frame flags the coordinate-space mismatch
    # uniformly rather than extracting wrong-scale 1:1 pixels on earlier frames.
    all_tracklet_boxes: list[tuple[float, float, float, float]] = [obs.bbox_xyxy for obs in observed_frames]
    all_tracklet_boxes.extend(
        obs.person_bbox_xyxy for obs in observed_frames if obs.person_bbox_xyxy is not None
    )

    cranial_hists: list[tuple[float, np.ndarray]] = []
    torso_hists: list[tuple[float, np.ndarray]] = []
    chassis_hists: list[tuple[float, np.ndarray]] = []
    qualities: list[float] = []
    last_rejection: str | None = "NO_MATCHING_FRAME_INDEX"
    rejection_counts: dict[str, int] = {}
    frames_with_keyframe = 0

    def _reject(reason: str) -> None:
        nonlocal last_rejection
        last_rejection = reason
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    def _crop_and_hist(
        frame_bgr: np.ndarray,
        bbox: tuple[float, float, float, float],
        x_rel: tuple[float, float],
        y_rel: tuple[float, float],
        sink: list[tuple[float, np.ndarray]],
    ) -> None:
        if native_wh is None and frame_bgr is not None and frame_bgr.size > 0 and frame_bgr.ndim == 3:
            img_h, img_w = frame_bgr.shape[:2]
            if any(
                b[2] > img_w + UNKNOWN_DIMS_TOLERANCE_PX or b[3] > img_h + UNKNOWN_DIMS_TOLERANCE_PX
                for b in all_tracklet_boxes
            ):
                _reject(REJ_UNKNOWN_DIMS_MISMATCH)
                return
        res: CropResult = crop_native_box(frame_bgr, bbox, native_wh, x_rel, y_rel)
        if res.crop is None:
            _reject(res.reason or "EMPTY_CROP")
            return
        ok, q, rej = compute_crop_quality(res.crop, config, min_dim_scale=min(res.scale_xy))
        if ok:
            sink.append((q, compute_hsv_achromatic_histogram(res.crop, config)))
            qualities.append(q)
        elif rej:
            _reject(rej)

    for obs in observed_frames:
        frame_bgr = frame_images_by_index.get(obs.frame_index)
        if frame_bgr is None:
            continue
        frames_with_keyframe += 1

        if is_motorcycle_tracklet:
            # 1. Motorcycle chassis core (lower-central 50% of motorcycle box)
            _crop_and_hist(frame_bgr, obs.bbox_xyxy, (0.20, 0.80), (0.35, 0.85), chassis_hists)
        else:
            # Rider or Orphan Rider tracklet: extract cranial cap core + upper torso
            _crop_and_hist(frame_bgr, obs.bbox_xyxy, (0.20, 0.80), (0.15, 0.75), cranial_hists)
            if obs.person_bbox_xyxy is not None:
                _crop_and_hist(
                    frame_bgr, obs.person_bbox_xyxy, (0.20, 0.80), (0.22, 0.62), torso_hists
                )

    def _weighted_avg(items: list[tuple[float, np.ndarray]]) -> list[float]:
        if not items:
            return []
        w_sum = sum(w for w, _ in items)
        if w_sum <= 1e-9:
            return []
        acc = sum(w * vec for w, vec in items) / w_sum
        s = float(np.sum(acc))
        if s > 1e-9:
            acc = acc / s
        return [round(float(x), 6) for x in acc]

    cranial_vec = _weighted_avg(cranial_hists)
    torso_vec = _weighted_avg(torso_hists)
    chassis_vec = _weighted_avg(chassis_hists)

    valid_count = len(cranial_hists) + len(torso_hists) + len(chassis_hists)
    if valid_count == 0:
        return AppearanceDescriptorRecord(
            available=False,
            unavailable_reason=last_rejection or "NO_VALID_CROPS_AFTER_QUALITY_GATE",
            valid_observed_crops_count=0,
            mean_crop_quality=0.0,
            observed_frames_total=len(observed_frames),
            observed_frames_with_keyframe=frames_with_keyframe,
            crop_rejection_counts=dict(rejection_counts),
        )

    mean_q = float(np.mean(qualities)) if qualities else 0.0
    if mean_q < config.min_valid_crop_quality:
        return AppearanceDescriptorRecord(
            available=False,
            unavailable_reason="MEAN_CROP_QUALITY_BELOW_THRESHOLD",
            valid_observed_crops_count=valid_count,
            mean_crop_quality=mean_q,
            observed_frames_total=len(observed_frames),
            observed_frames_with_keyframe=frames_with_keyframe,
            crop_rejection_counts=dict(rejection_counts),
        )

    return AppearanceDescriptorRecord(
        available=True,
        unavailable_reason=None,
        valid_observed_crops_count=valid_count,
        mean_crop_quality=mean_q,
        cranial_cap_histogram=cranial_vec,
        upper_torso_histogram=torso_vec,
        moto_chassis_histogram=chassis_vec,
        observed_frames_total=len(observed_frames),
        observed_frames_with_keyframe=frames_with_keyframe,
        crop_rejection_counts=dict(rejection_counts),
    )


def score_appearance_similarity(
    front_desc: AppearanceDescriptorRecord,
    back_desc: AppearanceDescriptorRecord,
    config: AppearanceConfig,
) -> tuple[bool, float | None, float, str | None, dict[str, Any]]:
    """
    Compare two `AppearanceDescriptorRecord` instances across available zones
    (cranial cap, upper torso, motorcycle chassis).
    Returns `(available, normalized_score, quality, unavailable_reason, diagnostics)`.
    """
    if not config.enabled:
        return False, None, 0.0, "CUE_DISABLED_IN_CONFIG", {}
    if not front_desc.available:
        return False, None, 0.0, f"FRONT_{front_desc.unavailable_reason}", {}
    if not back_desc.available:
        return False, None, 0.0, f"BACK_{back_desc.unavailable_reason}", {}

    zone_sims: dict[str, float] = {}
    zone_weights: dict[str, float] = {}

    sim_c = bhattacharyya_histogram_similarity(
        front_desc.cranial_cap_histogram, back_desc.cranial_cap_histogram
    )
    if sim_c is not None:
        zone_sims["cranial_cap"] = sim_c
        zone_weights["cranial_cap"] = config.weight_cranial_cap

    sim_t = bhattacharyya_histogram_similarity(
        front_desc.upper_torso_histogram, back_desc.upper_torso_histogram
    )
    if sim_t is not None:
        zone_sims["upper_torso"] = sim_t
        zone_weights["upper_torso"] = config.weight_upper_torso

    sim_m = bhattacharyya_histogram_similarity(
        front_desc.moto_chassis_histogram, back_desc.moto_chassis_histogram
    )
    if sim_m is not None:
        zone_sims["moto_chassis"] = sim_m
        zone_weights["moto_chassis"] = config.weight_moto_chassis

    if not zone_sims:
        return False, None, 0.0, "NO_COMMON_VALID_APPEARANCE_ZONES", {}

    w_total = sum(zone_weights.values())
    composite = sum(zone_sims[k] * zone_weights[k] for k in zone_sims) / max(1e-9, w_total)
    pair_quality = float(min(front_desc.mean_crop_quality, back_desc.mean_crop_quality))
    diagnostics = {
        "zone_similarities": {k: round(v, 4) for k, v in zone_sims.items()},
        "front_valid_crops": front_desc.valid_observed_crops_count,
        "back_valid_crops": back_desc.valid_observed_crops_count,
        "front_quality": round(front_desc.mean_crop_quality, 4),
        "back_quality": round(back_desc.mean_crop_quality, 4),
    }
    return True, float(np.clip(composite, 0.0, 1.0)), pair_quality, None, diagnostics
