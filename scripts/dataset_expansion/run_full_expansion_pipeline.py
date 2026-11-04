#!/usr/bin/env python3
"""run_full_expansion_pipeline.py: Master Autonomous Pipeline Orchestrator.

Orchestrates:
1. Waits for 02_annotate_dataset.py to complete all 1,841 images.
2. Runs Audit Pass 1 (03_audit_pass_1.py).
3. Runs Audit Pass 2 (04_audit_pass_2.py).
4. Runs Audit Pass 3 (05_audit_pass_3.py).
5. Runs Sequence-Isolated Splitting (06_create_training_split.py).
6. Runs Materialization and 18-Point Validation (07_materialize_and_validate.py).
7. Compiles Final Comprehensive Report & Charts (08_generate_final_report.py).
8. Renders visual previews across initial, audit_1, audit_2, audit_3, and training_ready.
"""
import os
import sys
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts/dataset_expansion"
EXPANSION_ROOT = PROJECT_ROOT / "data/helmet_dataset_expansion"
RAW_PRED_DIR = EXPANSION_ROOT / "annotations/raw_predictions"
PYTHON_EXE = "/opt/homebrew/bin/python3.11"
TOTAL_TARGET_IMAGES = 1841


def run_step(script_name: str, desc: str):
    print("\n" + "=" * 80)
    print(f"STARTING: {desc} ({script_name})")
    print("=" * 80)
    script_path = SCRIPTS_DIR / script_name
    t0 = time.time()
    res = subprocess.run([PYTHON_EXE, str(script_path)], capture_output=False, text=True)
    dur = time.time() - t0
    if res.returncode != 0:
        print(f"[FATAL ERROR] {script_name} failed with return code {res.returncode}")
        sys.exit(res.returncode)
    print(f"\n[COMPLETED] {desc} in {dur:.2f}s")


def wait_for_annotation_completion():
    print("=" * 80)
    print("MONITORING ANNOTATION PIPELINE PROGRESS (TARGET: 1,841 IMAGES)")
    print("=" * 80)

    last_count = 0
    while True:
        completed = len(list(RAW_PRED_DIR.glob("*.json")))
        if completed != last_count:
            pct = (completed / TOTAL_TARGET_IMAGES) * 100
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Annotation Progress: {completed}/{TOTAL_TARGET_IMAGES} images ({pct:.1f}%)")
            last_count = completed

        if completed >= TOTAL_TARGET_IMAGES:
            print("\nAll 1,841 images successfully annotated and saved to disk!")
            break

        time.sleep(15)


def main():
    print("=" * 80)
    print("MASTER AUTONOMOUS OVERNIGHT PIPELINE: MOTORCYCLE HELMET DATASET EXPANSION")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 1. Wait for annotation completion
    wait_for_annotation_completion()

    # 2. Audit Pass 1: Structural & Geometric Validation
    run_step("03_audit_pass_1.py", "Audit Pass 1: Structural & Geometric Validation")

    # 3. Audit Pass 2: Semantic & Multi-Rider Audit
    run_step("04_audit_pass_2.py", "Audit Pass 2: Semantic & Multi-Rider Quality Audit")

    # 4. Audit Pass 3: Final Independent Verification & QA Synthesis
    run_step("05_audit_pass_3.py", "Audit Pass 3: Final Independent Verification")

    # 5. Step 6: Sequence-Isolated Group Partitioning
    run_step("06_create_training_split.py", "Sequence-Isolated Dataset Split Generation")

    # 6. Step 7: Materialization & 18-Point Validation Suite
    run_step("07_materialize_and_validate.py", "Dataset Materialization & Automated Smoke Tests")

    # 7. Step 8: Final Master Report Compilation
    run_step("08_generate_final_report.py", "Compilation of DATASET_EXPANSION_FINAL_REPORT.md")

    # 8. Render visual previews for qualitative handoff
    print("\n" + "=" * 80)
    print("RENDERING QUALITATIVE VISUALIZATION SAMPLES ACROSS SPLITS")
    print("=" * 80)
    subprocess.run([
        PYTHON_EXE, str(SCRIPTS_DIR / "visualize_annotations.py"),
        "--img-dir", str(EXPANSION_ROOT / "training_ready/images/train"),
        "--lbl-dir", str(EXPANSION_ROOT / "training_ready/labels/train"),
        "--out-dir", str(EXPANSION_ROOT / "visualizations/training_ready_train"),
        "--limit", "30"
    ])
    subprocess.run([
        PYTHON_EXE, str(SCRIPTS_DIR / "visualize_annotations.py"),
        "--img-dir", str(EXPANSION_ROOT / "training_ready/images/val"),
        "--lbl-dir", str(EXPANSION_ROOT / "training_ready/labels/val"),
        "--out-dir", str(EXPANSION_ROOT / "visualizations/training_ready_val"),
        "--limit", "20"
    ])

    print("\n" + "=" * 80)
    print("MASTER OVERNIGHT EXPANSION MISSION COMPLETED SUCCESSFULLY!")
    print(f"Finished at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Final Status: READY FOR TRAINING")
    print(f"Training Dataset Location: {EXPANSION_ROOT / 'training_ready'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
