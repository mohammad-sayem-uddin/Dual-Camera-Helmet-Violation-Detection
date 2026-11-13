#!/usr/bin/env bash
set -euo pipefail

# Unified CPU-Safe Full Research Reproduction Script
# Replays cryptographic verification, Stage K (legacy & fixed tIoU), GAP-B, GAP-C,
# TASK-10 (bounded real video Cue C), TASK-11 (50-stream fusion ablation),
# TASK-12 (tIoU audit + v1.1 non-destruction check), frontend JS syntax check,
# and the complete pytest suite into an isolated non-destructive output directory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

OUT_ROOT="${1:-/tmp/research_repro_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_ROOT}"

# Resolve healthy Python 3.10+ interpreter (prefer explicit PYTHON_BIN, then Homebrew 3.11, then python3)
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="${PYTHON_BIN}"
elif [[ -x "/opt/homebrew/opt/python@3.11/bin/python3.11" ]]; then
  PY="/opt/homebrew/opt/python@3.11/bin/python3.11"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  echo "ERROR: No suitable python3 interpreter found." >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

LOG_FILE="${OUT_ROOT}/reproduction.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "========================================================================"
echo "Dual-Camera Helmet Violation Detection — Full Research Reproduction"
echo "========================================================================"
echo "Timestamp (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Repository Root: ${REPO_ROOT}"
echo "Git HEAD       : $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
echo "Python         : ${PY}"
echo "Output Root    : ${OUT_ROOT}"
echo ""
echo "[Scope Disclosures — Intentionally Excluded Stages]"
echo "  1. Phase 7/9 Kaggle T4 GPU training (frozen weights verified by SHA-256 below)."
echo "  2. Stage J full 50-stream video inference regeneration (reads frozen Stage J JSONLs)."
echo "  3. Blind human review execution (dual_camera_gt_v1.1 remains PENDING_HUMAN_REVIEW 0/92)."
echo "========================================================================"

echo ""
echo "[1/11] Checking Python environment and dependencies..."
"${PY}" - <<'PYEOF'
import importlib, platform, sys
required = ["torch", "ultralytics", "cv2", "numpy", "pandas", "yaml", "pytest", "scipy"]
missing = []
versions = {"python": sys.version.split()[0], "platform": platform.platform()}
for pkg in required:
    try:
        mod = importlib.import_module(pkg)
        versions[pkg] = getattr(mod, "__version__", "installed")
    except Exception as exc:
        missing.append((pkg, repr(exc)))
if missing:
    print("MISSING/BROKEN PACKAGES:", missing)
    sys.exit(1)
import torch
versions["cuda_available"] = bool(torch.cuda.is_available())
versions["mps_available"] = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
for k, v in versions.items():
    print(f"  {k}: {v}")
PYEOF

echo ""
echo "[2/11] Verifying cryptographic SHA-256 manifests and frozen weights..."
"${PY}" - <<'PYEOF'
import hashlib, json
from pathlib import Path

root = Path(".")

def check_manifest(manifest_rel: str, base_rel: str, expected_n: int) -> None:
    manifest = root / manifest_rel
    base = root / base_rel
    lines = [ln.strip() for ln in manifest.read_text().splitlines() if ln.strip()]
    assert len(lines) == expected_n, f"{manifest_rel}: expected {expected_n} entries, got {len(lines)}"
    for ln in lines:
        expected_hash, rel_file = ln.split(None, 1)
        target = base / rel_file.lstrip("*./") if not (base / rel_file).exists() else base / rel_file
        if not target.exists():
            # Fallback relative to manifest parent
            target = manifest.parent / rel_file
        assert target.exists(), f"Missing file in {manifest_rel}: {rel_file}"
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        assert actual == expected_hash, f"Checksum mismatch for {target}: {actual} != {expected_hash}"
    print(f"  OK ({len(lines)}/{expected_n}): {manifest_rel}")

check_manifest("data/dual_camera_test_v1/annotations/checksums/annotation_checksums.sha256", "data/dual_camera_test_v1/annotations", 39)
check_manifest("data/dual_camera_test_v1/releases/dual_camera_gt_v1.0_ai_assisted_audited/checksums/release_checksums.sha256", "data/dual_camera_test_v1/releases/dual_camera_gt_v1.0_ai_assisted_audited", 43)
check_manifest("results/phase12_integrated/checksums.sha256", "results/phase12_integrated", 8)

expected_weights = {
    "yolov8n.pt": "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36",
    "results/phase9_multiseed/exp_combined/seed_42/weights/best.pt": "0929a882667e0f7e2b631071ab218a2415cc1b816cdede7fb6812543d840b0e5",
    "results/exp_combined/weights/best.pt": "0929a882667e0f7e2b631071ab218a2415cc1b816cdede7fb6812543d840b0e5",
}
for rel, exp in expected_weights.items():
    p = root / rel
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    assert actual == exp, f"Weight mismatch {rel}: {actual} != {exp}"
    print(f"  OK weight {rel}: {actual[:16]}...")

v11_meta = json.loads((root / "data/dual_camera_test_v1/releases/dual_camera_gt_v1.1_blind_human/release_metadata.json").read_text())
assert v11_meta["human_verified"] is False
assert v11_meta["human_verification_status"] == "PENDING_HUMAN_REVIEW"
assert v11_meta["executed_human_units"] == 0
print(f"  OK v1.1 release untouched: {v11_meta['human_verification_status']}, executed={v11_meta['executed_human_units']}")
PYEOF

echo ""
echo "[3/11] Running Stage K evaluation (legacy tIoU parity check)..."
mkdir -p "${OUT_ROOT}/stagek_legacy"
"${PY}" scripts/reproducibility/evaluate.py --tiou-mode legacy --out "${OUT_ROOT}/stagek_legacy"

echo ""
echo "[4/11] Running Stage K evaluation (fixed Task-12 point-interval tIoU)..."
mkdir -p "${OUT_ROOT}/stagek_fixed"
"${PY}" scripts/reproducibility/evaluate.py --tiou-mode fixed --out "${OUT_ROOT}/stagek_fixed"

echo ""
echo "[5/11] Running GAP-B (TASK-06/06A) offline keyframe retention & evidence audit..."
mkdir -p "${OUT_ROOT}/gapb"
GAPB_EVAL_OUT_DIR="${OUT_ROOT}/gapb" "${PY}" scripts/reproducibility/evaluate_gapb_redesign.py

echo ""
echo "[6/11] Running GAP-C (TASK-09/09A/09B) motorcycle-anchored tracking & consensus ablation..."
mkdir -p "${OUT_ROOT}/gapc"
"${PY}" scripts/reproducibility/evaluate_gapc_tracking.py --out "${OUT_ROOT}/gapc"

echo ""
echo "[7/11] Running TASK-10 bounded real-video Cue C 720p/4K appearance verification..."
mkdir -p "${OUT_ROOT}/task10"
"${PY}" scripts/reproducibility/evaluate_task10_real_video_cue_c.py --out "${OUT_ROOT}/task10"

echo ""
echo "[8/11] Running TASK-11 50-stream dual-camera fusion & single-vs-dual ablation..."
mkdir -p "${OUT_ROOT}/task11"
"${PY}" scripts/reproducibility/evaluate_task11_fusion.py --out "${OUT_ROOT}/task11"

echo ""
echo "[9/11] Running TASK-12 tIoU boundary & blind GT v1.1 non-destruction audit..."
mkdir -p "${OUT_ROOT}/task12"
"${PY}" scripts/reproducibility/evaluate_task12_audit.py --out "${OUT_ROOT}/task12"

echo ""
echo "[10/11] Checking review dashboard frontend JavaScript syntax..."
if command -v node >/dev/null 2>&1; then
  node --check src/review_dashboard/static/app.js
  echo "  OK: node --check src/review_dashboard/static/app.js ($(node --version))"
else
  echo "  SKIPPED: node not installed on PATH"
fi

echo ""
echo "[11/11] Running complete pytest regression & integration test suite..."
"${PY}" -m pytest -q tests/

echo ""
echo "[Post] Writing consolidated reproduction manifest and enforcing gate checks..."
OUT_ROOT_ENV="${OUT_ROOT}" "${PY}" - <<'PYEOF'
import json, os, sys
from pathlib import Path

out = Path(os.environ["OUT_ROOT_ENV"])
gapb = json.loads((out / "gapb/gapb_evaluation_results.json").read_text())
gapc = json.loads((out / "gapc/gapc_ablation_results.json").read_text())
t10 = json.loads((out / "task10/task10_cue_c_results.json").read_text())
t11 = json.loads((out / "task11/task11_fusion_results.json").read_text())
t12 = json.loads((out / "task12/task12_tiou_comparison.json").read_text())

gates = {
    "gapb_item_level_100pct": (
        gapb["gap_b4_stagek_item_level_rider_association"]["retained_motorcycles"]["coverage"] == 1.0
        and gapb["gap_b4_stagek_item_level_rider_association"]["retained_motorcycles"]["f1"] == 1.0
    ),
    "gapc_all_acceptance_criteria_passed": bool(gapc["all_acceptance_criteria_passed"]),
    "task10_all_checks_passed": bool(t10["all_checks_passed"]),
    "task11_all_checks_passed": bool(t11["all_checks_passed"]),
    "task12_all_checks_passed": bool(t12["all_checks_passed"]),
}

all_passed = all(gates.values())
manifest = {
    "status": "PASSED" if all_passed else "FAILED",
    "output_root": str(out),
    "stagek_legacy_event_f1": t12["legacy_eval_summary"]["f1"],
    "stagek_fixed_event_f1": t12["fixed_eval_summary"]["f1"],
    "gates": gates,
    **gates,
}
(out / "reproduction_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
if not all_passed:
    failed = [k for k, v in gates.items() if not v]
    print(f"ERROR: Reproduction gate failure(s): {failed}", file=sys.stderr)
    sys.exit(1)
PYEOF

echo ""
echo "========================================================================"
echo "FULL REPRODUCTION COMPLETED SUCCESSFULLY -> ${OUT_ROOT}"
echo "========================================================================"
