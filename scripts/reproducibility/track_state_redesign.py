"""TASK-04 amended track-level helmet evaluation (additive; Stage K legacy untouched).

Re-scores the frozen GT v1.0 against existing Stage J `tracks_final.json` artifacts using
the genuine-observation track-state redesign. No video inference, no model loading.

Per-track states are taken from exported fields when present (new pipeline runs), else
recomputed from `rider_pts_history` / `pts_history` (legacy Stage J artifacts) with the
exact same rules as `src/tracking/camera_tracker.py` (`HelmetEvidenceAccumulator`).

Usage (from repo root):
    python scripts/reproducibility/track_state_redesign.py [--out results/phase12_track_state_redesign]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from src.tracking.camera_tracker import (  # noqa: E402
    HelmetEvidenceAccumulator,
    derive_ema_helmet_state,
)

REL = os.path.join(REPO, "data/dual_camera_test_v1/releases/dual_camera_gt_v1.0_ai_assisted_audited")
SJ = os.path.join(REPO, "results/phase12_stagej")
PAIRS = [f"pair_{i:03d}" for i in range(1, 26)]
CAMS = ("front", "back")
METHODS = ("ema_final", "last_coupled", "majority", "conf_weighted", "primary")
EMA_ALPHA = 0.35
EXPORTED = {
    "ema_final": "track_helmet_state_ema_final",
    "last_coupled": "track_helmet_state_last_coupled",
    "majority": "track_helmet_state_majority",
    "conf_weighted": "track_helmet_state_conf_weighted",
    "primary": "track_helmet_state",
}


def replay_from_history(hist, ema_alpha: float = EMA_ALPHA) -> dict:
    """Recompute exported fields from (frame, pts, bbox, class, conf) history (fallback)."""
    acc = HelmetEvidenceAccumulator()
    ema = None
    for h in sorted(hist, key=lambda x: x[0]):
        cls, conf = h[3], float(h[4])
        if cls not in ("helmet", "no_helmet"):
            continue
        val = conf if cls == "helmet" else -conf
        ema = val if ema is None else ema_alpha * val + (1 - ema_alpha) * ema
        p_h = round((max(-1.0, min(1.0, ema)) + 1.0) / 2.0, 4)
        acc.record(cls, conf, h[0], h[1], {"helmet": p_h, "no_helmet": round(1.0 - p_h, 4)})
    return acc.export({"helmet": 0.0, "no_helmet": 0.0})


def load_gt() -> dict:
    gt = {}
    for pf in sorted(glob.glob(REL + "/pairs/*/pair_*_gt.json")):
        pr = json.load(open(pf))["pair"]
        pair = pr["pair_id"]
        for cam in CAMS:
            ca = pr["camera_annotations"][cam]
            for mt in ca["motorcycle_tracks"]:
                for st in mt["rider_subtracks"]:
                    gt[(pair, cam, st["subtrack_id"])] = (st["gt_helmet_state"], "moto_subtrack")
            for ot in ca["orphan_rider_tracks"]:
                gt[(pair, cam, ot["track_id"])] = (ot["gt_helmet_state"], "orphan")
    return gt


def load_pred() -> tuple[dict, str]:
    """(pair,cam,id) -> {state fields..., legacy_ac_max}. Returns also the source mode."""
    pred, modes = {}, set()
    for pair in PAIRS:
        for cam in CAMS:
            tf = json.load(open(f"{SJ}/{pair}/{pair}_{cam}_tracks_final.json"))
            for mt in tf["moto_tracks"]:
                rph = mt.get("rider_pts_history", {})
                rs = mt.get("rider_subtracks") or {}
                for st in rs.values() if isinstance(rs, dict) else rs:
                    pred[(pair, cam, st["subtrack_id"])] = _entry(st, rph.get(str(st["slot_index"]), []), modes)
            for ot in tf["orphan_tracks"]:
                oid = ot.get("orphan_track_id") or ot.get("track_id")
                pred[(pair, cam, oid)] = _entry(ot, ot.get("pts_history", []), modes)
    return pred, ("exported_fields" if modes == {"exported"} else "pts_history_fallback" if modes == {"fallback"} else "mixed")


def _entry(rec: dict, hist: list, modes: set) -> dict:
    ac = rec.get("aggregated_confidence") or {}
    if "track_helmet_state" in rec:
        modes.add("exported")
        e = {k: rec[v] for k, v in EXPORTED.items()}
        e["n"] = rec["coupled_observations_count"]
        e["switches"] = rec["coupled_state_switches"]
    else:
        modes.add("fallback")
        f = replay_from_history(hist)
        e = {k: f[v] for k, v in EXPORTED.items()}
        e["ema_final"] = derive_ema_helmet_state(ac, tie_break_helmet=True)  # legacy Stage K rule
        e["n"] = f["coupled_observations_count"]
        e["switches"] = f["coupled_state_switches"]
        e["_lc_ac"] = f["last_coupled_aggregated_confidence"]
    e["legacy_ac_max"] = max(ac.values()) if ac else 0.0
    return e


def _reason(method: str, e: dict) -> str:
    if e["n"] == 0:
        return "ZERO_GENUINE_OBSERVATIONS"
    if method == "ema_final":
        return "WINDOW_FLUSH_ZERO_MASS" if e["legacy_ac_max"] == 0 else "BELOW_0.5_CONFIDENCE"
    if method == "last_coupled":
        return "BELOW_0.5_OR_EXACT_TIE"
    return "EXACT_TIE"


def score(rows: list[dict], method: str) -> dict:
    conf, reasons = collections.Counter(), collections.Counter()
    for r in rows:
        p = r["pred"][method]
        conf[(r["gt"].upper(), p)] += 1
        if p == "UNCERTAIN":
            reasons[_reason(method, r["pred"])] += 1
    total = len(rows)
    correct = sum(v for (g, p), v in conf.items() if g == p)
    abst = sum(v for (g, p), v in conf.items() if p == "UNCERTAIN")
    decided = total - abst
    return {
        "total_eligible_gt_tracks": total, "decided": decided, "abstentions": abst,
        "coverage": round(decided / total, 4) if total else None,
        "conditional_accuracy": round(correct / decided, 4) if decided else None,
        "end_to_end_agreement": round(correct / total, 4) if total else None,
        "correct": correct, "wrong": decided - correct,
        "confusion": {f"{g}->{p}": v for (g, p), v in sorted(conf.items())},
        "abstention_reasons": dict(reasons),
    }


def evaluate() -> dict:
    gt, (pred, mode) = load_gt(), load_pred()
    rows = []
    for k, (g, kind) in gt.items():
        if g in ("helmet", "no_helmet"):
            p = pred.get(k)
            if p is None:  # key missing -> legacy default UNCERTAIN, 0 genuine observations
                p = {m: "UNCERTAIN" for m in METHODS}
                p.update(n=0, switches=0, legacy_ac_max=0.0)
            rows.append({"key": k, "gt": g, "kind": kind, "pred": p})
    groups = {
        "all": rows,
        "single_observation_K1": [r for r in rows if r["pred"]["n"] == 1],
        "multi_observation_K>=2": [r for r in rows if r["pred"]["n"] >= 2],
        "motorcycle_subtracks": [r for r in rows if r["kind"] == "moto_subtrack"],
        "orphan_tracks": [r for r in rows if r["kind"] == "orphan"],
        "legacy_window_flushed_tail_gap>=10_proxy": [r for r in rows if r["pred"]["legacy_ac_max"] == 0],
        "legacy_not_flushed": [r for r in rows if r["pred"]["legacy_ac_max"] > 0],
        "has_state_transition_switches>=1": [r for r in rows if r["pred"]["switches"] >= 1],
        "no_state_transition_switches=0": [r for r in rows if r["pred"]["switches"] == 0],
    }
    out = {"source_mode": mode, "gt_release": "dual_camera_gt_v1.0_ai_assisted_audited (read-only)",
           "stage_j_artifacts": "results/phase12_stagej (read-only)",
           "note": "Implementation re-score of existing artifacts; thresholds not tuned. "
                   "Tail-gap subgroup uses legacy-window-flush as a proxy (track_frames.jsonl is not "
                   "present in this repo).",
           "groups": {g: {m: score(rs, m) for m in METHODS} for g, rs in groups.items()}}
    out["_rows"] = rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "results/phase12_track_state_redesign"))
    a = ap.parse_args()
    res = evaluate()
    rows = res.pop("_rows")
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "helmet_track_level_redesign.json"), "w"), indent=1)
    with open(os.path.join(a.out, "per_track_states.csv"), "w") as f:
        f.write("pair,camera,track_id,kind,gt," + ",".join(METHODS) + ",n_genuine_obs,state_switches\n")
        for r in sorted(rows, key=lambda r: r["key"]):
            p = r["pred"]
            f.write(",".join([*r["key"], r["kind"], r["gt"]] + [p[m] for m in METHODS] + [str(p["n"]), str(p["switches"])]) + "\n")
    for m in METHODS:
        s = res["groups"]["all"][m]
        print(f"{m:14s} decided={s['decided']:3d} abst={s['abstentions']:2d} cov={s['coverage']} "
              f"cond_acc={s['conditional_accuracy']} e2e={s['end_to_end_agreement']}")


if __name__ == "__main__":
    main()
