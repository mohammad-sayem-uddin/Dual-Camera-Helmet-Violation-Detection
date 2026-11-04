#!/usr/bin/env python3
"""
Collect and unify Phase 9 Multi-Seed Results across Seeds 42, 123, 2026.
Integrates audited Seed 42 with fresh Kaggle execution outputs (Seeds 123, 2026).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compute_phase9_statistics import compute_multiseed_analysis


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def normalize_eval_dict(evals: dict) -> dict:
    norm = {}
    for k, v in evals.items():
        if "new" in k:
            norm["new_test"] = v
        elif "old" in k:
            norm["old_test"] = v
        elif "comb" in k:
            norm["combined_test"] = v
        else:
            norm[k] = v
    return norm


def main():
    root = Path(".")
    phase9_dir = root / "results/phase9_multiseed"
    
    # 1. Load Audited Seed 42 for EXP-NEW
    exp_new_seed42_file = root / "results/exp_new/exp_new_evaluation_summary.json"
    assert exp_new_seed42_file.exists(), f"Missing {exp_new_seed42_file}"
    raw_new_42 = json.loads(exp_new_seed42_file.read_text())
    seed42_new = {
        "experiment": "EXP-NEW",
        "seed": 42,
        "evaluations": normalize_eval_dict(raw_new_42["evaluations"])
    }
    
    # 2. Load Audited Seed 42 for EXP-COMBINED
    exp_comb_seed42_file = root / "results/exp_combined/exp_combined_results_summary.json"
    assert exp_comb_seed42_file.exists(), f"Missing {exp_comb_seed42_file}"
    raw_comb_42 = json.loads(exp_comb_seed42_file.read_text())
    seed42_comb = {
        "experiment": "EXP-COMBINED",
        "seed": 42,
        "evaluations": normalize_eval_dict(raw_comb_42["evaluations"])
    }
    
    # Check for downloaded Kaggle outputs
    new_outputs_file = phase9_dir / "kaggle_outputs_exp_new/exp_new_multiseed_summary.json"
    comb_outputs_file = phase9_dir / "kaggle_outputs_exp_comb/exp_combined_multiseed_summary.json"
    
    new_multiseed = {"42": seed42_new}
    comb_multiseed = {"42": seed42_comb}
    
    if new_outputs_file.exists():
        data_new = json.loads(new_outputs_file.read_text())
        for seed_str, val in data_new.items():
            val["evaluations"] = normalize_eval_dict(val["evaluations"])
            new_multiseed[seed_str] = val
        print(f"Loaded EXP-NEW seeds: {list(new_multiseed.keys())}")
    else:
        print(f"Waiting for {new_outputs_file}...")
        
    if comb_outputs_file.exists():
        data_comb = json.loads(comb_outputs_file.read_text())
        for seed_str, val in data_comb.items():
            val["evaluations"] = normalize_eval_dict(val["evaluations"])
            comb_multiseed[seed_str] = val
        print(f"Loaded EXP-COMBINED seeds: {list(comb_multiseed.keys())}")
    else:
        print(f"Waiting for {comb_outputs_file}...")

    # If all 3 seeds are available for both, run full analysis
    if len(new_multiseed) >= 3 and len(comb_multiseed) >= 3:
        print("All 3 seeds loaded for both experiments! Running full statistical analysis...")
        stats = compute_multiseed_analysis(
            new_results_by_seed=new_multiseed,
            comb_results_by_seed=comb_multiseed,
            output_dir=phase9_dir,
        )
        print("Done!")
    else:
        print(f"Current seed status: EXP-NEW has {len(new_multiseed)}/3, EXP-COMBINED has {len(comb_multiseed)}/3.")

if __name__ == "__main__":
    main()
