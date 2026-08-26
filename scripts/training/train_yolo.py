#!/usr/bin/env python3
"""Configurable, reproducible YOLOv8 training entry point for research experiments."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import platform
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import ensure_parent, research_path
from src.utils.yaml_loader import dump_yaml, load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 and record complete experiment metadata.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration and paths without starting training")
    parser.add_argument("--device", default=None, help="Override device (e.g. 'mps', 'cuda:0', 'cpu', 'auto')")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    config_path = research_path(args.config)
    if not config_path.exists():
        parser.error(f"Experiment config not found: {config_path}")

    config = load_yaml(config_path)

    required_keys = ("experiment_id", "model", "dataset_yaml", "imgsz", "batch", "epochs", "seed")
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        parser.error(f"Config is missing required keys: {', '.join(missing_keys)}")

    # CLI Overrides
    if args.device is not None:
        config["device"] = args.device
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch is not None:
        config["batch"] = args.batch

    experiment_id = config["experiment_id"]
    experiment_dir = research_path(Path("experiments") / experiment_id)
    dataset_yaml = research_path(config["dataset_yaml"])

    if not dataset_yaml.exists():
        parser.error(f"Dataset YAML does not exist: {dataset_yaml}")

    if experiment_dir.exists() and not args.dry_run:
        parser.error(f"Experiment directory already exists: {experiment_dir}. Use a new experiment_id.")

    print("=" * 70)
    print(f"EXPERIMENT INITIALIZATION: {experiment_id}")
    print("=" * 70)
    print(f"Model Architecture:       {config['model']}")
    print(f"Dataset Config:           {dataset_yaml}")
    print(f"Image Resolution:         {config['imgsz']}x{config['imgsz']}")
    print(f"Batch Size:               {config['batch']}")
    print(f"Epochs:                   {config['epochs']}")
    print(f"Random Seed:              {config['seed']}")
    print(f"Patience:                 {config.get('patience', 100)}")
    print(f"Target Output Directory:  {experiment_dir}")
    print("=" * 70)

    if args.dry_run:
        print("[DRY-RUN] Configuration and dataset paths validated successfully. Ready for training.")
        return

    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "config.yaml").write_text(dump_yaml(config), encoding="utf-8")

    # Import YOLO inside main so CLI help and dry-run are fast without torch import overhead
    try:
        from ultralytics import YOLO
        import torch
    except ImportError as e:
        sys.exit(f"Error: Ultralytics/PyTorch not installed. Please run pip install -r requirements.txt.\nDetails: {e}")

    # Resolve hardware device
    req_device = config.get("device", "auto")
    if req_device in (None, "auto", ""):
        if torch.cuda.is_available():
            resolved_device = "0"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            resolved_device = "cpu"
    else:
        resolved_device = req_device

    # Build Ultralytics training arguments
    train_kwargs = {
        "data": str(dataset_yaml),
        "imgsz": config["imgsz"],
        "batch": config["batch"],
        "epochs": config["epochs"],
        "seed": config["seed"],
        "deterministic": config.get("deterministic", True),
        "project": str(experiment_dir),
        "name": "training",
        "patience": config.get("patience", 100),
        "optimizer": config.get("optimizer", "auto"),
        "device": resolved_device,
        "save": True,
        "plots": True,
    }

    if config.get("learning_rate") is not None:
        train_kwargs["lr0"] = config["learning_rate"]
    if config.get("lrf") is not None:
        train_kwargs["lrf"] = config["lrf"]
    if config.get("weight_decay") is not None:
        train_kwargs["weight_decay"] = config["weight_decay"]
    if config.get("momentum") is not None:
        train_kwargs["momentum"] = config["momentum"]

    augmentation_params = config.get("augmentation_params", {})
    if not isinstance(augmentation_params, dict):
        parser.error("augmentation_params must be a mapping of Ultralytics options.")
    train_kwargs.update(augmentation_params)

    # Record environment metadata
    env_metadata = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    }

    metadata = {
        "experiment_id": experiment_id,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "environment": env_metadata,
        "config": config,
        "training_arguments": train_kwargs,
    }

    start_time = time.perf_counter()
    print(f"Starting YOLOv8 training on device '{train_kwargs['device']}'...")
    
    model = YOLO(config["model"])
    results = model.train(**train_kwargs)
    
    duration = round(time.perf_counter() - start_time, 2)
    metadata["duration_seconds"] = duration
    metadata["finished_at_utc"] = datetime.now(UTC).isoformat()

    # Save run metadata
    (experiment_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Mirror best and last checkpoints to models/checkpoints/{experiment_id}/
    weights_dir = experiment_dir / "training" / "weights"
    checkpoint_dest = ensure_parent(research_path(Path("models/checkpoints") / experiment_id))
    checkpoint_dest.mkdir(parents=True, exist_ok=True)

    if (weights_dir / "best.pt").exists():
        shutil.copy2(weights_dir / "best.pt", checkpoint_dest / "best.pt")
        print(f"Saved best weights checkpoint to: {checkpoint_dest / 'best.pt'}")
    if (weights_dir / "last.pt").exists():
        shutil.copy2(weights_dir / "last.pt", checkpoint_dest / "last.pt")
        print(f"Saved last weights checkpoint to: {checkpoint_dest / 'last.pt'}")

    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE: {experiment_id}")
    print(f"Duration: {duration}s")
    print(f"Experiment Artifacts: {experiment_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
