"""Environment and Dependency Readiness Test Suite."""
import os
import shutil
import subprocess
import pytest


def test_core_dependencies_import():
    """Verify all required deep learning and computer vision libraries import without error."""
    import torch
    import torchvision
    import ultralytics
    import cv2
    import numpy as np
    import pandas as pd
    import PIL
    import scipy
    import soundfile
    import yaml
    import easyocr

    assert torch.__version__ is not None
    assert ultralytics.__version__ is not None
    assert cv2.__version__ is not None
    assert easyocr.__version__ is not None


def test_ffmpeg_and_ffprobe_available():
    """Verify ffmpeg and ffprobe binaries are installed and accessible."""
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")

    assert ffmpeg_path is not None, "ffmpeg binary not found in PATH"
    assert ffprobe_path is not None, "ffprobe binary not found in PATH"

    res = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "ffmpeg version" in res.stdout


def test_apple_silicon_mps_support():
    """Verify PyTorch detects Apple Silicon Metal Performance Shaders (MPS)."""
    import torch
    assert torch.backends.mps.is_available(), "PyTorch MPS backend should be available on Apple Silicon M1."


def test_disk_space_sufficiency():
    """Verify sufficient disk space is available for local testing (> 5 GB)."""
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024 ** 3)
    assert free_gb >= 5.0, f"Insufficient disk space: {free_gb:.2f} GB available, need >= 5 GB."
