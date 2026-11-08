"""Model Adapters Unit Test Suite."""
from pathlib import Path
import numpy as np
import pytest

from src.models.helmet_detector_adapter import HelmetDetectorAdapter
from src.models.plate_detector_adapter import HeuristicPlateDetector, PretrainedPlateDetectorAdapter
from src.models.ocr_adapter import EasyOCRAdapter


def test_helmet_detector_initialization_and_classes():
    """Verify helmet detector loads checkpoint and exposes correct classes."""
    adapter = HelmetDetectorAdapter(device="cpu")
    info = adapter.get_info()

    assert info["verified"] is True
    assert info["classes"] == {0: "no_helmet", 1: "helmet"}
    assert Path(info["weights_path"]).exists()


def test_helmet_detector_inference_on_dummy_frame():
    """Verify single-frame inference returns standardized dictionary output."""
    adapter = HelmetDetectorAdapter(device="cpu")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)

    result = adapter.predict(dummy)

    assert "detections" in result
    assert "latency_ms" in result
    assert "device" in result
    assert isinstance(result["detections"], list)
    assert result["latency_ms"] >= 0.0


def test_heuristic_plate_detector():
    """Verify heuristic plate ROI extractor correctly crops lower region of vehicle box."""
    detector = HeuristicPlateDetector()
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
    veh_box = [500.0, 200.0, 900.0, 600.0]  # w=400, h=400

    plates = detector.detect_plates(frame, veh_box)

    assert len(plates) == 1
    p = plates[0]
    assert "bbox" in p
    assert "crop" in p
    assert p["crop"].shape[0] > 0
    assert p["crop"].shape[1] > 0
    # Expected: x1 = 500 + 400*0.2 = 580; x2 = 900 - 400*0.2 = 820
    # y1 = 200 + 400*0.65 = 460; y2 = 200 + 400*0.95 = 580
    assert p["bbox"] == [580.0, 460.0, 820.0, 580.0]


def test_pretrained_plate_detector_placeholder():
    """Verify placeholder interface raises NotImplementedError as intended."""
    detector = PretrainedPlateDetectorAdapter()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(NotImplementedError):
        detector.detect_plates(frame, [10, 10, 50, 50])


def test_easyocr_adapter_text_cleaning_and_read():
    """Verify EasyOCR text cleaning and character extraction."""
    ocr = EasyOCRAdapter(use_gpu=False)

    # Test text cleaning method
    dirty = "Dhaka Metro-Ha 12-3456!@#"
    cleaned = ocr.clean_plate_text(dirty)
    assert cleaned == "DHAKA METRO-HA 12-3456"

    # Test reading blank patch (returns empty text without crashing)
    blank = np.zeros((40, 100, 3), dtype=np.uint8)
    res = ocr.read_text(blank)
    assert "text" in res
    assert "confidence" in res
    assert "latency_ms" in res
