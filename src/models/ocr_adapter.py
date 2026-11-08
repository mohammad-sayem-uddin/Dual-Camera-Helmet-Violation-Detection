"""OCR Adapter Interface and Implementations (EasyOCR primary, Tesseract fallback)."""
from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from typing import Any
import numpy as np

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

try:
    import pytesseract
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False


class BaseOCRAdapter(ABC):
    """Abstract base class for OCR text extraction."""

    @abstractmethod
    def read_text(
        self,
        image_or_crop: np.ndarray,
        conf_threshold: float = 0.20,
    ) -> dict[str, Any]:
        """
        Extract text from an image or cropped plate region.
        
        Args:
            image_or_crop: BGR numpy array.
            conf_threshold: Minimum character/token confidence threshold.
            
        Returns:
            Dictionary:
                - text: Extracted and cleaned text string
                - raw_results: List of raw OCR detections (bbox, text, conf)
                - confidence: Mean confidence of recognized text
                - latency_ms: Float execution time in milliseconds
        """
        pass


class EasyOCRAdapter(BaseOCRAdapter):
    """
    Primary OCR adapter using EasyOCR (CRAFT + CRNN with PyTorch).
    Runs locally on CPU or Apple Silicon MPS.
    """

    def __init__(
        self,
        languages: list[str] | None = None,
        use_gpu: bool = False,
    ) -> None:
        if not HAS_EASYOCR:
            raise ImportError("easyocr package is required. Install via pip install easyocr.")

        self.languages = languages or ["en"]
        self.use_gpu = use_gpu
        self._reader: easyocr.Reader | None = None

    @property
    def reader(self) -> easyocr.Reader:
        if self._reader is None:
            self._reader = easyocr.Reader(self.languages, gpu=self.use_gpu)
        return self._reader

    def clean_plate_text(self, text: str) -> str:
        """Strip non-alphanumeric noise, keeping letters, numbers, and hyphens."""
        # Clean whitespace and unwanted symbols
        cleaned = re.sub(r"[^A-Za-z0-9\- ]", "", text).strip()
        cleaned = re.sub(r"\s+", " ", cleaned).upper()
        return cleaned

    def read_text(
        self,
        image_or_crop: np.ndarray,
        conf_threshold: float = 0.20,
    ) -> dict[str, Any]:
        if image_or_crop is None or image_or_crop.size == 0:
            return {"text": "", "raw_results": [], "confidence": 0.0, "latency_ms": 0.0}

        t0 = time.perf_counter()
        raw_results = self.reader.readtext(image_or_crop)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        filtered_tokens = []
        scores = []
        raw_boxes = []

        for bbox, text, score in raw_results:
            if score >= conf_threshold:
                cleaned_tok = self.clean_plate_text(text)
                if cleaned_tok:
                    filtered_tokens.append(cleaned_tok)
                    scores.append(float(score))
                    raw_boxes.append({
                        "bbox": [[int(pt[0]), int(pt[1])] for pt in bbox],
                        "text": cleaned_tok,
                        "confidence": round(float(score), 4),
                    })

        full_text = " ".join(filtered_tokens)
        mean_conf = float(np.mean(scores)) if scores else 0.0

        return {
            "text": full_text,
            "raw_results": raw_boxes,
            "confidence": round(mean_conf, 4),
            "latency_ms": round(latency_ms, 2),
            "engine": "EasyOCR",
            "device": "GPU" if self.use_gpu else "CPU",
        }


class TesseractOCRAdapter(BaseOCRAdapter):
    """
    Backup OCR adapter using Tesseract via pytesseract.
    Requires system tesseract binary.
    """

    def __init__(self, tesseract_cmd: str | None = None) -> None:
        self.available = False
        if HAS_PYTESSERACT:
            if tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            try:
                pytesseract.get_tesseract_version()
                self.available = True
            except Exception:
                self.available = False

    def read_text(
        self,
        image_or_crop: np.ndarray,
        conf_threshold: float = 0.20,
    ) -> dict[str, Any]:
        if not self.available:
            return {
                "text": "",
                "raw_results": [],
                "confidence": 0.0,
                "latency_ms": 0.0,
                "error": "Tesseract system binary not installed or found in PATH.",
            }

        t0 = time.perf_counter()
        # Process with pytesseract
        import cv2
        gray = cv2.cvtColor(image_or_crop, cv2.COLOR_BGR2GRAY) if len(image_or_crop.shape) == 3 else image_or_crop
        text = pytesseract.image_to_string(gray, config="--psm 7")
        latency_ms = (time.perf_counter() - t0) * 1000.0

        cleaned = re.sub(r"[^A-Za-z0-9\- ]", "", text).strip().upper()
        return {
            "text": cleaned,
            "raw_results": [{"text": cleaned, "confidence": 0.50}],
            "confidence": 0.50,
            "latency_ms": round(latency_ms, 2),
            "engine": "Tesseract",
            "device": "CPU",
        }
