"""Stage J validation: per-pair gates + manifest reconciliation + Pass 1 deltas.

Usage: python validate_stagej.py <stagej_out_dir>
Writes work/stageJ/validation_results.json and prints a compact summary.

Gates per pair (ALL must pass):
  G1 summary exists, status == COMPLETE
  G2 summary validation.all_passed == True (8/8 incl. all_passed flag)
  G3 frames_decoded (summary coverage) == Phase 12C forensic count
     (frozen package frame_coverage_manifest.csv rows per pair/camera),
     and == frames_expected_container
  G4 detections JSONL lines == frames_decoded, per camera
  G5 all 13 per-pair artifacts present (11 pair_* + pair_status + pair_summary)
  G6 sync diagnostics carries drift_ms AND q5_overlap_lookup_stats (GAP-1/4)
  G7 summary inputs_sha256 weights == recorded hashes (staged weights)
Also: run_manifest.json reconciles 25/25 COMPLETE; Pass 1 per-category
detection deltas per pair/camera (flag |delta| > 2% of the Pass 1 count,
with a floor: only flag when the absolute delta >= 5 boxes as well, so
tiny-count categories are not noise-flagged); totals for events and
association outcomes aggregated across pairs.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/home/hatch/workspace/research_helmet")
OUT = Path(sys.argv[1])
PAIRS = [f"pair_{i:03d}" for i in range(1, 26)]
CATS = ["motorcycle", "person", "helmet", "no_helmet"]
EXPECTED_WEIGHTS = {
    "weights_yolov8n_sha256": "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36",
    "weights_phase9_best_sha256": "0929a882667e0f7e2b631071ab218a2415cc1b816cdede7fb6812543d840b0e5",
}

# ---- forensic frame counts (independent source: frozen package manifest)
forensic: dict[tuple[str, str], int] = Counter()
with open(ROOT / "repo/data/dual_camera_test_v1/annotations/frame_coverage_manifest.csv") as f:
    for row in csv.DictReader(f):
        forensic[(row["pair_id"], row["camera"])] += 1

# ---- Pass 1 detection totals per pair/camera/category
def pass1_totals(pair: str, cam: str) -> Counter:
    p = ROOT / f"work/pass1/{pair}/{pair}_{cam}_detections.jsonl"
    tot = Counter()
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for d in rec["detections"]:
                tot[d["class_name"]] += 1
    return tot


def stagej_totals(path: Path) -> tuple[int, Counter]:
    tot = Counter()
    lines = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines += 1
            b = json.loads(line)
            tot["motorcycle"] += b["num_motorcycles"]
            tot["person"] += b["num_persons"]
            tot["helmet"] += b["num_helmet"]
            tot["no_helmet"] += b["num_no_helmet"]
    return lines, tot


ARTIFACT_SUFFIXES = [
    "association.json", "back_coverage.csv", "back_detections.jsonl",
    "back_track_frames.jsonl", "back_tracks_final.json", "events.json",
    "front_coverage.csv", "front_detections.jsonl",
    "front_track_frames.jsonl", "front_tracks_final.json",
    "sync_diagnostics.json",
]

results: dict = {"pairs": {}, "flags": []}
agg_stagej = Counter()
agg_pass1 = Counter()
agg_events = 0
agg_events_by_type = Counter()
agg_assoc = Counter()
agg_rider = Counter()
agg_orphan = Counter()
gate_pass = 0

for pair in PAIRS:
    d = OUT / pair
    pr: dict = {"gates": {}}
    ok_box = [True]

    def gate(name, cond, detail=""):
        pr["gates"][name] = {"pass": bool(cond), "detail": detail}
        ok_box[0] = ok_box[0] and bool(cond)

    summ_path = d / "pair_summary.json"
    if not summ_path.is_file():
        gate("G1_summary_complete", False, "pair_summary.json missing")
        results["pairs"][pair] = pr
        continue
    s = json.loads(summ_path.read_text())
    gate("G1_summary_complete", s.get("status") == "COMPLETE", s.get("status", ""))
    v = s.get("validation", {})
    gate("G2_validation_8of8", v.get("all_passed") is True
         and all(v.get(k) is True for k in (
             "coverage_complete_front", "coverage_complete_back",
             "detections_pts_match_table", "association_outcome_counts_reconcile",
             "association_results_reference_known_tracklets",
             "events_cover_all_moto_tracks", "sync_diagnostics_all_q_fields")),
         json.dumps({k: v.get(k) for k in sorted(v)}))
    cov = s.get("coverage", {})
    det_lines, det_tot = {}, {}
    for cam in ("front", "back"):
        exp = forensic[(pair, cam)]
        c = cov.get(cam, {})
        jl = d / f"{pair}_{cam}_detections.jsonl"
        lines, tot = stagej_totals(jl) if jl.is_file() else (-1, Counter())
        det_lines[cam], det_tot[cam] = lines, tot
        gate(f"G3_frames_{cam}", c.get("frames_decoded") == exp
             and c.get("frames_expected_container") == exp,
             f"decoded={c.get('frames_decoded')} expected_container={c.get('frames_expected_container')} forensic={exp}")
        gate(f"G4_jsonl_lines_{cam}", lines == exp, f"lines={lines} forensic={exp}")
        agg_stagej.update({(cam, cat): tot[cat] for cat in CATS})
    need = [f"{pair}_{sfx}" for sfx in ARTIFACT_SUFFIXES] + ["pair_status.json", "pair_summary.json"]
    missing = [n for n in need if not (d / n).is_file()]
    gate("G5_artifacts_13", not missing, f"missing={missing}")
    diag_path = d / f"{pair}_sync_diagnostics.json"
    if diag_path.is_file():
        diag = json.loads(diag_path.read_text())
        gate("G6_diag_drift_q5", "drift_ms" in diag and "q5_overlap_lookup_stats" in diag,
             f"drift_ms={diag.get('drift_ms')} q5_samples={diag.get('q5_overlap_lookup_stats', {}).get('n_samples')}")
        pr["drift_ms"] = diag.get("drift_ms")
    else:
        gate("G6_diag_drift_q5", False, "diagnostics file missing")
    ish = s.get("inputs_sha256", {})
    gate("G7_staged_weights", all(ish.get(k) == w for k, w in EXPECTED_WEIGHTS.items()),
         json.dumps({k: (ish.get(k) or "")[:12] for k in EXPECTED_WEIGHTS}))
    pr["status"] = s.get("status")
    pr["wall_time_sec"] = s.get("wall_time_sec")
    pr["peak_rss_mb"] = s.get("peak_rss_mb")
    cnt = s.get("counts", {})
    pr["events_total"] = cnt.get("events_total")
    agg_events += cnt.get("events_total") or 0
    agg_events_by_type.update(cnt.get("events_by_type", {}))
    for src, agg in (("association_outcome_counts", agg_assoc),
                     ("rider_outcome_counts", agg_rider),
                     ("orphan_outcome_counts", agg_orphan)):
        agg.update(cnt.get(src, {}))
    # Pass 1 deltas
    pr["pass1_deltas"] = {}
    for cam in ("front", "back"):
        p1 = pass1_totals(pair, cam)
        agg_pass1.update({(cam, cat): p1[cat] for cat in CATS})
        for cat in CATS:
            delta = det_tot[cam][cat] - p1[cat]
            pr["pass1_deltas"][f"{cam}_{cat}"] = delta
            if p1[cat] > 0:
                pct = abs(delta) / p1[cat] * 100
                if pct > 2.0 and abs(delta) >= 5:
                    results["flags"].append(
                        f"{pair} {cam} {cat}: stagej={det_tot[cam][cat]} pass1={p1[cat]} delta={delta} ({pct:.1f}%)")
    pr["all_gates_pass"] = ok_box[0]
    gate_pass += 1 if ok_box[0] else 0
    results["pairs"][pair] = pr

man_path = OUT / "run_manifest.json"
manifest = json.loads(man_path.read_text()) if man_path.is_file() else {}
mpairs = {p["pair_id"]: p for p in manifest.get("pairs", [])}
results["manifest"] = {
    "present": bool(manifest),
    "counts": manifest.get("counts", {}),
    "n_pairs_listed": len(mpairs),
    "all_complete": len(mpairs) == 25
    and all(p.get("status") == "COMPLETE" for p in mpairs.values()),
    "input_hashes": manifest.get("input_hashes", {}),
    "config": manifest.get("config", {}),
}
results["gate_pass_pairs"] = gate_pass
results["aggregate"] = {
    "stagej_detections": {f"{c}_{cat}": agg_stagej[(c, cat)] for c in ("front", "back") for cat in CATS},
    "pass1_detections": {f"{c}_{cat}": agg_pass1[(c, cat)] for c in ("front", "back") for cat in CATS},
    "deltas": {f"{c}_{cat}": agg_stagej[(c, cat)] - agg_pass1[(c, cat)] for c in ("front", "back") for cat in CATS},
    "events_total": agg_events,
    "events_by_type": dict(agg_events_by_type),
    "association_outcomes": dict(agg_assoc),
    "rider_outcomes": dict(agg_rider),
    "orphan_outcomes": dict(agg_orphan),
    "forensic_totals": {"front": sum(v for (p, c), v in forensic.items() if c == "front"),
                        "back": sum(v for (p, c), v in forensic.items() if c == "back")},
}
(ROOT / "work/stageJ/validation_results.json").write_text(json.dumps(results, indent=1, sort_keys=True))
print(f"gate-pass pairs: {gate_pass}/25")
print(f"manifest: {results['manifest']['counts']} all_complete={results['manifest']['all_complete']}")
print(f"events total: {agg_events} by_type={dict(agg_events_by_type)}")
print(f"assoc={dict(agg_assoc)} rider={dict(agg_rider)} orphan={dict(agg_orphan)}")
print(f"aggregate deltas vs Pass 1: {results['aggregate']['deltas']}")
print(f"flags (>2% and >=5 boxes): {len(results['flags'])}")
for fl in results["flags"]:
    print("  FLAG:", fl)
fails = [p for p, r in results["pairs"].items() if not r.get("all_gates_pass")]
print("pairs failing gates:", fails)
