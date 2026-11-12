"""Explicit coordinate-space contract for bounding boxes, stored images and borders (GAP-B1/B2).

Contract
--------
1. Every `ObservedFrameRecord` bounding box lives in the coordinate space of the
   *native decoded frame* of its camera (e.g. 3840x2160). Records are never rescaled.
2. Any stored image (appearance keyframe) may have a different size. Its actual
   `(height, width)` is read from the array itself; the transform from native to
   stored coordinates is `sx = img_w / native_w`, `sy = img_h / native_h`, applied
   per axis (anisotropic scaling is handled, never assumed away).
3. When the native dimensions are unknown the stored image is treated as native
   *only if every box is consistent with the image bounds*; otherwise the crop is
   rejected with an explicit reason. There is no silent 1920x1080 fallback.
4. Crops are clipped to the image; fully out-of-frame, invalid or zero-area
   regions are rejected with an explicit reason instead of returning pixels from an
   unrelated image location.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

# Explicit rejection reasons (appear in `AppearanceDescriptorRecord.unavailable_reason`
# and `crop_rejection_counts`).
REJ_EMPTY_IMAGE = "EMPTY_IMAGE"
REJ_INVALID_BOX = "INVALID_BOX"
REJ_OUT_OF_FRAME = "BOX_OUT_OF_FRAME"
REJ_ZERO_AREA = "ZERO_AREA_CROP"
REJ_UNKNOWN_DIMS_MISMATCH = "COORDINATE_SPACE_MISMATCH_UNKNOWN_NATIVE_DIMENSIONS"

# Boxes may exceed the image by this many pixels before an unknown-dimension
# (assumed native) image is declared inconsistent with the box coordinate space.
UNKNOWN_DIMS_TOLERANCE_PX = 2.0


def normalize_dimensions(dims: Any) -> tuple[int, int] | None:
    """Validate `(width, height)`; returns None when missing, raises on malformed input."""
    if dims is None:
        return None
    try:
        w, h = dims
        w_i, h_i = int(w), int(h)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"frame dimensions must be (width, height), got {dims!r}") from exc
    if w_i <= 0 or h_i <= 0 or w_i != w or h_i != h:
        raise ValueError(f"frame dimensions must be positive integers, got {dims!r}")
    return (w_i, h_i)


def scale_factors(native_wh: tuple[int, int], image_hw: tuple[int, int]) -> tuple[float, float]:
    """Per-axis scale from native coordinates to stored-image coordinates."""
    nw, nh = native_wh
    ih, iw = image_hw
    return float(iw) / float(nw), float(ih) / float(nh)


@dataclass
class CropResult:
    crop: np.ndarray | None
    reason: str | None
    scale_xy: tuple[float, float] = (1.0, 1.0)
    image_box_xyxy: tuple[int, int, int, int] | None = None

    @property
    def ok(self) -> bool:
        return self.crop is not None


def crop_native_box(
    image_bgr: np.ndarray | None,
    bbox_xyxy: tuple[float, float, float, float],
    native_wh: tuple[int, int] | None,
    x_rel_range: tuple[float, float] = (0.0, 1.0),
    y_rel_range: tuple[float, float] = (0.0, 1.0),
) -> CropResult:
    """Crop a relative sub-region of a native-coordinate box from a stored image."""
    if image_bgr is None or image_bgr.size == 0 or image_bgr.ndim != 3:
        return CropResult(None, REJ_EMPTY_IMAGE)
    img_h, img_w = image_bgr.shape[:2]
    try:
        x1, y1, x2, y2 = (float(c) for c in bbox_xyxy)
    except (TypeError, ValueError):
        return CropResult(None, REJ_INVALID_BOX)
    if not all(math.isfinite(c) for c in (x1, y1, x2, y2)):
        return CropResult(None, REJ_INVALID_BOX)
    if (x2 - x1) < 2.0 or (y2 - y1) < 2.0:
        return CropResult(None, REJ_INVALID_BOX)

    if native_wh is None:
        # Stored image is assumed native only if the box is consistent with it.
        if x2 > img_w + UNKNOWN_DIMS_TOLERANCE_PX or y2 > img_h + UNKNOWN_DIMS_TOLERANCE_PX:
            return CropResult(None, REJ_UNKNOWN_DIMS_MISMATCH)
        sx = sy = 1.0
        nw, nh = img_w, img_h
    else:
        nw, nh = native_wh
        sx, sy = scale_factors(native_wh, (img_h, img_w))

    # Entirely outside the native frame: do not clamp into an unrelated border strip.
    if x2 <= 0.0 or y2 <= 0.0 or x1 >= nw or y1 >= nh:
        return CropResult(None, REJ_OUT_OF_FRAME, (sx, sy))

    bw, bh = x2 - x1, y2 - y1
    rx0, rx1 = x_rel_range
    ry0, ry1 = y_rel_range
    # Sub-region in native coordinates, then mapped into stored-image coordinates.
    ax1 = (x1 + rx0 * bw) * sx
    ax2 = (x1 + rx1 * bw) * sx
    ay1 = (y1 + ry0 * bh) * sy
    ay2 = (y1 + ry1 * bh) * sy
    ix1 = int(round(max(0.0, min(float(img_w), ax1))))
    ix2 = int(round(max(0.0, min(float(img_w), ax2))))
    iy1 = int(round(max(0.0, min(float(img_h), ay1))))
    iy2 = int(round(max(0.0, min(float(img_h), ay2))))
    if ix2 <= ix1 or iy2 <= iy1:
        return CropResult(None, REJ_ZERO_AREA, (sx, sy), (ix1, iy1, ix2, iy2))
    return CropResult(image_bgr[iy1:iy2, ix1:ix2], None, (sx, sy), (ix1, iy1, ix2, iy2))


def is_box_near_border(
    bbox_xyxy: tuple[float, float, float, float],
    frame_wh: tuple[int, int],
    margin_px: float,
) -> bool:
    """Pixel-margin border test; box and frame MUST share one coordinate space."""
    w, h = frame_wh
    x1, y1, x2, y2 = bbox_xyxy
    return bool(
        x1 <= margin_px or y1 <= margin_px or x2 >= (w - margin_px) or y2 >= (h - margin_px)
    )


# A box may overshoot the frame slightly (detector jitter); beyond this it is evidence
# that the box and the declared frame are in different coordinate spaces.
BORDER_MISMATCH_TOLERANCE_PX = 8.0


def boxes_consistent_with_frame(
    boxes: list[tuple[float, float, float, float]],
    frame_wh: tuple[int, int],
    tolerance_px: float = BORDER_MISMATCH_TOLERANCE_PX,
) -> bool:
    w, h = frame_wh
    return all(
        b[0] >= -tolerance_px
        and b[1] >= -tolerance_px
        and b[2] <= w + tolerance_px
        and b[3] <= h + tolerance_px
        for b in boxes
    )
