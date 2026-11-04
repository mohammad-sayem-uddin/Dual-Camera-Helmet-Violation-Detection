#!/usr/bin/env python3
"""Multi-seed orchestration pipeline for Phase 4 statistical validation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import ensure_parent, research_path


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_experiment_seed(exp_base: str, config_file: str, dataset_yaml: str, seed: int, dry_run: bool = False) -> None:
    exp_id = exp_base if seed == 42 else f"{exp_base}_seed_{seed}"
    exp_dir = research_path(f"experiments/{exp_id}")
    ckpt_dir = research_path(f"models/checkpoints/{exp_id}")
    metrics_dir = research_path("results/metrics")
    val_json = metrics_dir / f"{exp_id}-val.json"
    test_json = metrics_dir / f"{exp_id}-test.json"
    
    print("\n" + "=" * 80)
    print(f"PIPELINE STEP: {exp_id} (Seed {seed})")
    print("=" * 80)

    # 1. Check if valid completed run exists
    best_pt = ckpt_dir / "best.pt"
    if not best_pt.exists() and (exp_dir / "training" / "weights" / "best.pt").exists():
        best_pt = exp_dir / "training" / "weights" / "best.pt"

    if best_pt.exists() and val_json.exists() and test_json.exists():
        print(f"✅ Reusing existing valid run for {exp_id}:")
        print(f"   Checkpoint: {best_pt} (SHA256: {compute_sha256(best_pt)})")
        print(f"   Val Metrics:  {val_json}")
        print(f"   Test Metrics: {test_json}")
        return

    if dry_run:
        print(f"[DRY-RUN] Would train and evaluate {exp_id} on seed {seed}.")
        return

    # 2. Train model
    is_resuming = (exp_dir / "training" / "weights" / "last.pt").exists()
    print(f"\n>>> {'Resuming' if is_resuming else 'Starting'} Training for {exp_id} on seed {seed}...")
    cmd_train = [
        sys.executable, "scripts/training/train_yolo.py",
        "--config", config_file,
        "--experiment-id", exp_id,
        "--seed", str(seed)
    ]
    if is_resuming:
        cmd_train.append("--resume")
    t0 = time.perf_counter()
    subprocess.run(cmd_train, check=True)
    duration = time.perf_counter() - t0
    print(f">>> Training completed in {duration/3600:.2f} hours ({duration:.1f}s).")

    # 3. Freeze checkpoint to models/checkpoints/{exp_id}/best.pt
    src_best = exp_dir / "training" / "weights" / "best.pt"
    dst_best = ckpt_dir / "best.pt"
    ensure_parent(dst_best)
    shutil.copy2(src_best, dst_best)
    print(f">>> Checkpoint saved: {dst_best} (SHA256: {compute_sha256(dst_best)})")

    # 4. Standardized Validation Evaluation
    print(f"\n>>> Running Validation Evaluation for {exp_id}...")
    cmd_val = [
        sys.executable, "scripts/evaluation/evaluate_yolo.py",
        "--weights", str(dst_best),
        "--data", dataset_yaml,
        "--split", "val",
        "--name", f"{exp_id}-val"
    ]
    subprocess.run(cmd_val, check=True)

    # 5. Standardized Test Evaluation
    print(f"\n>>> Running Test Evaluation for {exp_id}...")
    cmd_test = [
        sys.executable, "scripts/evaluation/evaluate_yolo.py",
        "--weights", str(dst_best),
        "--data", dataset_yaml,
        "--split", "test",
        "--name", f"{exp_id}-test"
    ]
    subprocess.run(cmd_test, check=True)

    # 6. Training History Analysis
    results_csv = exp_dir / "training" / "results.csv"
    if results_csv.exists():
        print(f"\n>>> Running Training History Analysis for {exp_id}...")
        cmd_ana = [
            sys.executable, "scripts/analysis/analyze_training.py",
            "--results-csv", str(results_csv),
            "--experiment-id", exp_id
        ]
        subprocess.run(cmd_ana, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4 multi-seed statistical validation.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026, 7, 99], help="Seed list")
    parser.add_argument("--dry-run", action="store_true", help="Validate pipeline steps without training")
    parser.add_argument("--only-stats", action="store_true", help="Only run statistical analysis across existing runs")
    args = parser.parse_args()

    experiments = [
        ("EXP-003", "configs/experiments/exp_003_size_matched_baseline.yaml", "configs/dataset_random_matched.yaml"),
        ("EXP-002-FIXED", "configs/experiments/exp_002_group_split_fixed.yaml", "configs/dataset_group_split_fixed.yaml")
    ]

    print("=" * 80)
    print("PHASE 4 MULTI-SEED VALIDATION ORCHESTRATOR")
    print(f"Target Seeds: {args.seeds}")
    print(f"Total Conditions: {len(experiments)} exps × {len(args.seeds)} seeds = {len(experiments) * len(args.seeds)} runs")
    print("=" * 80)

    if not args.only_stats:
        for seed in args.seeds:
            for exp_base, cfg, dyaml in experiments:
                run_experiment_seed(exp_base, cfg, dyaml, seed, dry_run=args.dry_run)

    # Run multi-seed statistical aggregation
    print("\n" + "=" * 80)
    print("COMPUTING FINAL MULTI-SEED STATISTICAL REPORT")
    print("=" * 80)
    cmd_stats = [
        sys.executable, "scripts/analysis/compute_statistical_comparison.py",
        "--seeds", *[str(s) for s in args.seeds],
        "--output", "results/tables/THREE_WAY_SCIENTIFIC_COMPARISON.md"
    ]
    subprocess.run(cmd_stats, check=True)


if __name__ == "__main__":
    main()
