"""Check environment and hardware status."""
import platform
import cv2
import matplotlib
import numpy as np
import pandas as pd
import torch
import ultralytics
import yaml

print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("Ultralytics:", ultralytics.__version__)
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Matplotlib:", matplotlib.__version__)
print("PyYAML:", yaml.__version__)
mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
print("MPS Available:", mps_ok)
print("CUDA Available:", torch.cuda.is_available())
if mps_ok:
    # Test a small MPS tensor allocation
    x = torch.ones(2, 2, device="mps")
    print("MPS Tensor Test: Success, allocated on", x.device)
