"""Video Frame Processing and Visual Annotation Engine with PIL and OpenCV fallback."""
from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# Color palette in RGB
STATE_COLORS: dict[str, tuple[int, int, int]] = {
    "HELMET_CONFIRMED": (46, 204, 113),      # Green
    "NO_HELMET_CONFIRMED": (231, 76, 60),    # Red
    "NO_HELMET_SUSPECTED": (243, 156, 18),   # Amber/Orange
    "UNKNOWN": (149, 165, 166),              # Neutral Gray
    "TEMPORARILY_OCCLUDED": (155, 89, 182),  # Purple
    "TRACK_LOST": (52, 73, 94),              # Dark Slate
}


class FrameProcessor:
    """
    Renders temporal verification bounding boxes, trajectory trails,
    confidence readouts, and violation banners onto video frames.
    """

    def __init__(self, font_size: int = 14) -> None:
        self.font_size = font_size

    def annotate_frame(
        self,
        frame_path_or_array: str | Path | Any,
        frame_data: dict[str, Any],
        output_path: str | Path | None = None,
    ) -> Any:
        """
        Draws active tracks, temporal states, and violation alerts onto an image/frame.
        
        Args:
            frame_path_or_array: Image path (str/Path) or raw array.
            frame_data: Dictionary output from TemporalVerificationEngine.process_frame().
            output_path: Optional destination path to save annotated image.
            
        Returns:
            Annotated image object.
        """
        if not HAS_PIL:
            # Fallback when PIL is missing
            return None

        # Load image with PIL
        if isinstance(frame_path_or_array, (str, Path)):
            img = Image.open(str(frame_path_or_array)).convert("RGB")
        elif HAS_CV2 and isinstance(frame_path_or_array, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(frame_path_or_array, cv2.COLOR_BGR2RGB))
        elif hasattr(frame_path_or_array, "convert"):
            img = frame_path_or_array.convert("RGB")
        else:
            # Create a blank canvas if invalid
            img = Image.new("RGB", (640, 480), color=(30, 30, 30))

        draw = ImageDraw.Draw(img)

        active_tracks = frame_data.get("active_tracks", [])
        new_alerts = frame_data.get("new_alerts", [])
        f_idx = frame_data.get("frame_index", 0)

        # 1. Draw each active track
        for track in active_tracks:
            bbox = track.get("bbox", [0, 0, 0, 0])
            state = track.get("state", "UNKNOWN")
            tid = track.get("track_id", 0)
            conf_dict = track.get("aggregated_confidence", {})
            p_no_h = conf_dict.get("no_helmet", 0.0)
            p_h = conf_dict.get("helmet", 0.0)

            color = STATE_COLORS.get(state, (200, 200, 200))
            x1, y1, x2, y2 = bbox

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Draw label banner
            label = f"ID:{tid} | {state}"
            sublabel = f"no_h:{p_no_h:.2f} h:{p_h:.2f}"

            # Label box background
            text_y1 = max(0.0, y1 - 28)
            draw.rectangle([x1, text_y1, x1 + 180, text_y1 + 26], fill=color)
            draw.text((x1 + 4, text_y1 + 2), label, fill=(255, 255, 255))
            draw.text((x1 + 4, text_y1 + 14), sublabel, fill=(255, 255, 255))

        # 2. Draw violation alert banner if new alerts present
        if new_alerts:
            alert_text = f"!!! VIOLATION ALERT DETECTED (Frame {f_idx}) !!!"
            draw.rectangle([0, 0, img.width, 36], fill=(231, 76, 60))
            draw.text((20, 10), alert_text, fill=(255, 255, 255))

        # 3. Save if output path requested
        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(p))

        return img
