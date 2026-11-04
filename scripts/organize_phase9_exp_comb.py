#!/usr/bin/env python3
"""Organize EXP-COMBINED multi-seed artifacts and verify SHA-256 hashes."""
import hashlib
import json
from pathlib import Path
import shutil

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    root = Path(".")
    p9_dir = root / "results/phase9_multiseed"
    exp_comb_target = p9_dir / "exp_combined"
    exp_comb_target.mkdir(parents=True, exist_ok=True)
    
    # 1. Copy Seed 42 from Phase 7
    s42_target = exp_comb_target / "seed_42"
    s42_target.mkdir(parents=True, exist_ok=True)
    (s42_target / "weights").mkdir(exist_ok=True)
    
    s42_best = root / "results/exp_combined/weights/best.pt"
    if s42_best.exists():
        shutil.copy2(s42_best, s42_target / "weights/best.pt")
    shutil.copy2(root / "results/exp_combined/exp_combined_results_summary.json", s42_target / "seed_summary.json")
    if (root / "results/exp_combined/results.csv").exists():
        shutil.copy2(root / "results/exp_combined/results.csv", s42_target / "results.csv")
    if (root / "results/exp_combined/BoxPR_curve.png").exists():
        shutil.copy2(root / "results/exp_combined/BoxPR_curve.png", s42_target / "BoxPR_curve.png")
    if (root / "results/exp_combined/confusion_matrix.png").exists():
        shutil.copy2(root / "results/exp_combined/confusion_matrix.png", s42_target / "confusion_matrix.png")
    print(f"Copied Seed 42 artifacts to {s42_target}!")
        
    # 2. Copy Seed 123 and Seed 2026 from downloaded Kaggle outputs
    kaggle_out = p9_dir / "kaggle_outputs_exp_comb/exp_combined"
    for s in ["seed_123", "seed_2026"]:
        src_dir = kaggle_out / s
        dst_dir = exp_comb_target / s
        if src_dir.exists():
            if dst_dir.exists():
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)
            print(f"Copied {s} artifacts to {dst_dir}!")
            
    # 3. Create SHA-256 Manifest
    manifest = {}
    for seed_dir in sorted(exp_comb_target.iterdir()):
        if seed_dir.is_dir():
            best_pt = seed_dir / "weights/best.pt"
            if best_pt.exists():
                manifest[f"EXP-COMBINED_{seed_dir.name}_best.pt"] = {
                    "path": str(best_pt),
                    "size_bytes": best_pt.stat().st_size,
                    "sha256": sha256_file(best_pt)
                }
                
    manifest_path = p9_dir / "exp_combined_checkpoints_sha256.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Manifest written to {manifest_path} with {len(manifest)} checkpoints:")
    for k, v in manifest.items():
        print(f"  {k}: {v['sha256'][:16]}... ({v['size_bytes'] / 1e6:.2f} MB)")

if __name__ == "__main__":
    main()
