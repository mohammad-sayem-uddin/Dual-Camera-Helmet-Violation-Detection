#!/usr/bin/env python3
"""build_smoke_test_notebook.py

Updates smoke test notebook to dynamically discover mounted dataset locations in /kaggle/input.
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Motorcycle Helmet Violation Detection — 2-Epoch Pre-Training Smoke Test\n",
            "**Purpose**: Verify GPU environment, CUDA visibility, dataset mounts, 2-epoch YOLOv8n convergence, checkpoint serialization, and per-epoch runtime estimation."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Install Dependencies & Hardware Verification\n",
            "import subprocess\n",
            "import sys\n",
            "\n",
            "print('=' * 60)\n",
            "print('1. INSTALLING ULTRALYTICS & DEPENDENCIES')\n",
            "print('=' * 60)\n",
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics', 'pandas'], check=True)\n",
            "\n",
            "import os\n",
            "import time\n",
            "import json\n",
            "import torch\n",
            "import ultralytics\n",
            "from pathlib import Path\n",
            "\n",
            "print(f'Python version: {sys.version.split()[0]}')\n",
            "print(f'PyTorch version: {torch.__version__}')\n",
            "print(f'Ultralytics version: {ultralytics.__version__}')\n",
            "print(f'CUDA Available: {torch.cuda.is_available()}')\n",
            "\n",
            "gpu_info = {}\n",
            "if torch.cuda.is_available():\n",
            "    device_count = torch.cuda.device_count()\n",
            "    device_name = torch.cuda.get_device_name(0)\n",
            "    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)\n",
            "    gpu_info = {\n",
            "        'cuda_available': True,\n",
            "        'device_count': device_count,\n",
            "        'device_name': device_name,\n",
            "        'total_vram_gb': round(total_vram_gb, 2)\n",
            "    }\n",
            "    print(f'GPU Device Count: {device_count}')\n",
            "    print(f'GPU Model: {device_name}')\n",
            "    print(f'Total VRAM: {total_vram_gb:.2f} GB')\n",
            "else:\n",
            "    gpu_info = {'cuda_available': False, 'device_name': 'CPU', 'total_vram_gb': 0.0}\n",
            "    print('WARNING: CUDA is not available! Training on CPU.')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. Dataset Mount Discovery & Path Resolution\n",
            "print('=' * 60)\n",
            "print('2. DATASET MOUNT DISCOVERY & AUDIT')\n",
            "print('=' * 60)\n",
            "\n",
            "input_dir = Path('/kaggle/input')\n",
            "print('Scanning /kaggle/input structure:')\n",
            "for root, dirs, files in os.walk(input_dir):\n",
            "    level = root.replace(str(input_dir), '').count(os.sep)\n",
            "    if level <= 3:\n",
            "        indent = ' ' * 4 * level\n",
            "        print(f'{indent}{Path(root).name}/ ({len(dirs)} dirs, {len(files)} files)')\n",
            "\n",
            "# Locate New Dataset and Old Dataset directories wherever mounted\n",
            "new_ds_path = None\n",
            "old_ds_path = None\n",
            "\n",
            "for p in input_dir.rglob('*'):\n",
            "    if p.is_dir() and 'helmet-dataset-expansion-v2' in p.name.lower():\n",
            "        new_ds_path = p\n",
            "    elif p.is_dir() and 'old-dataset-v1-fixed' in p.name.lower():\n",
            "        old_ds_path = p\n",
            "\n",
            "# Fallback checks\n",
            "if not new_ds_path:\n",
            "    for p in input_dir.glob('**/*'):\n",
            "        if (p / 'images/train').exists():\n",
            "            new_ds_path = p\n",
            "            break\n",
            "\n",
            "if not old_ds_path:\n",
            "    for p in input_dir.glob('**/*'):\n",
            "        if (p / 'train/images').exists():\n",
            "            old_ds_path = p\n",
            "            break\n",
            "\n",
            "print(f'Resolved New Dataset Path: {new_ds_path}')\n",
            "print(f'Resolved Old Dataset Path: {old_ds_path}')\n",
            "\n",
            "dataset_audit = {}\n",
            "if new_ds_path and new_ds_path.exists():\n",
            "    n_tr = len(list((new_ds_path / 'images/train').glob('*.jpg')))\n",
            "    n_va = len(list((new_ds_path / 'images/val').glob('*.jpg')))\n",
            "    n_te = len(list((new_ds_path / 'images/test').glob('*.jpg')))\n",
            "    dataset_audit['new_dataset'] = {\n",
            "        'path': str(new_ds_path),\n",
            "        'train_images': n_tr,\n",
            "        'val_images': n_va,\n",
            "        'test_images': n_te,\n",
            "        'total_images': n_tr + n_va + n_te\n",
            "    }\n",
            "    print(f'New Dataset: {n_tr} train, {n_va} val, {n_te} test images (Total: {n_tr + n_va + n_te})')\n",
            "\n",
            "if old_ds_path and old_ds_path.exists():\n",
            "    o_tr = len(list((old_ds_path / 'train/images').glob('*.jpg')))\n",
            "    o_va = len(list((old_ds_path / 'val/images').glob('*.jpg')))\n",
            "    o_te = len(list((old_ds_path / 'test/images').glob('*.jpg')))\n",
            "    dataset_audit['old_dataset'] = {\n",
            "        'path': str(old_ds_path),\n",
            "        'train_images': o_tr,\n",
            "        'val_images': o_va,\n",
            "        'test_images': o_te,\n",
            "        'total_images': o_tr + o_va + o_te\n",
            "    }\n",
            "    print(f'Old Dataset: {o_tr} train, {o_va} val, {o_te} test images (Total: {o_tr + o_va + o_te})')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. Configure YOLOv8 Dataset YAML\n",
            "work_dir = Path('/kaggle/working')\n",
            "smoke_yaml_path = work_dir / 'smoke_data.yaml'\n",
            "\n",
            "# Point to whichever valid dataset was found\n",
            "train_root = new_ds_path if new_ds_path else old_ds_path\n",
            "if new_ds_path:\n",
            "    smoke_yaml_content = f\"\"\"path: {new_ds_path}\n",
            "train: images/train\n",
            "val: images/val\n",
            "test: images/test\n",
            "\n",
            "names:\n",
            "  0: no_helmet\n",
            "  1: helmet\n",
            "\n",
            "nc: 2\n",
            "\"\"\"\n",
            "else:\n",
            "    smoke_yaml_content = f\"\"\"path: {old_ds_path}\n",
            "train: train/images\n",
            "val: val/images\n",
            "test: test/images\n",
            "\n",
            "names:\n",
            "  0: no_helmet\n",
            "  1: helmet\n",
            "\n",
            "nc: 2\n",
            "\"\"\"\n",
            "\n",
            "with open(smoke_yaml_path, 'w') as f:\n",
            "    f.write(smoke_yaml_content)\n",
            "\n",
            "print(f'Generated smoke test data.yaml at {smoke_yaml_path}:\\n{smoke_yaml_content}')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 4. Run 2-Epoch Smoke Test\n",
            "from ultralytics import YOLO\n",
            "\n",
            "print('=' * 60)\n",
            "print('4. EXECUTING 2-EPOCH YOLOv8n SMOKE TEST')\n",
            "print('=' * 60)\n",
            "\n",
            "start_time = time.time()\n",
            "model = YOLO('yolov8n.pt')\n",
            "\n",
            "results = model.train(\n",
            "    data=str(smoke_yaml_path),\n",
            "    epochs=2,\n",
            "    imgsz=640,\n",
            "    batch=32,\n",
            "    device=0 if torch.cuda.is_available() else 'cpu',\n",
            "    seed=42,\n",
            "    project='/kaggle/working/smoke_test_run',\n",
            "    name='smoke_test',\n",
            "    exist_ok=True,\n",
            "    verbose=True\n",
            ")\n",
            "\n",
            "total_duration = time.time() - start_time\n",
            "print('=' * 60)\n",
            "print(f'Smoke Test Completed in {total_duration:.2f} seconds ({total_duration/60:.2f} minutes).')\n",
            "print('=' * 60)\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 5. Checkpoint, Metrics & Artifact Verification\n",
            "run_dir = Path('/kaggle/working/smoke_test_run/smoke_test')\n",
            "weights_dir = run_dir / 'weights'\n",
            "best_pt = weights_dir / 'best.pt'\n",
            "last_pt = weights_dir / 'last.pt'\n",
            "results_csv = run_dir / 'results.csv'\n",
            "\n",
            "checkpoint_verification = {\n",
            "    'best_pt_exists': best_pt.exists(),\n",
            "    'best_pt_bytes': best_pt.stat().st_size if best_pt.exists() else 0,\n",
            "    'last_pt_exists': last_pt.exists(),\n",
            "    'last_pt_bytes': last_pt.stat().st_size if last_pt.exists() else 0,\n",
            "    'results_csv_exists': results_csv.exists(),\n",
            "}\n",
            "\n",
            "print(f'Checkpoint best.pt exists: {checkpoint_verification[\"best_pt_exists\"]} ({checkpoint_verification[\"best_pt_bytes\"]} bytes)')\n",
            "print(f'Checkpoint last.pt exists: {checkpoint_verification[\"last_pt_exists\"]} ({checkpoint_verification[\"last_pt_bytes\"]} bytes)')\n",
            "print(f'Results CSV exists: {checkpoint_verification[\"results_csv_exists\"]}')\n",
            "\n",
            "epoch_metrics = []\n",
            "if results_csv.exists():\n",
            "    import pandas as pd\n",
            "    df_res = pd.read_csv(results_csv)\n",
            "    df_res.columns = df_res.columns.str.strip()\n",
            "    print('\\nTraining Metrics Summary:')\n",
            "    print(df_res[['epoch', 'train/box_loss', 'train/cls_loss', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']])\n",
            "    epoch_metrics = df_res.to_dict(orient='records')\n",
            "\n",
            "smoke_summary = {\n",
            "    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),\n",
            "    'gpu_info': gpu_info,\n",
            "    'dataset_audit': dataset_audit,\n",
            "    'smoke_test_runtime_seconds': round(total_duration, 2),\n",
            "    'time_per_epoch_seconds': round(total_duration / 2.0, 2),\n",
            "    'checkpoint_verification': checkpoint_verification,\n",
            "    'epoch_metrics': epoch_metrics\n",
            "}\n",
            "\n",
            "with open(work_dir / 'smoke_test_summary.json', 'w') as f:\n",
            "    json.dump(smoke_summary, f, indent=2)\n",
            "\n",
            "print(f'\\nSaved smoke_test_summary.json to {work_dir / \"smoke_test_summary.json\"}')\n"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = NOTEBOOK_DIR / "kaggle_smoke_test.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

out_smoke = NOTEBOOK_DIR / "smoke_test/kaggle_smoke_test.ipynb"
with open(out_smoke, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Updated smoke test notebook at: {out_path} and {out_smoke}")
