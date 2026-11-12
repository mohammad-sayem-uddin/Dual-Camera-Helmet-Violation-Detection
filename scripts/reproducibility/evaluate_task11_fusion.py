"""Offline TASK-11 Dual-Camera Fusion & Fair Single-vs-Dual Ablation (all 50 streams).

Evaluates `src/fusion/` on CPU over cached Stage J detections and PTS tables
across all 25 pairs (`pair_001`..`pair_025`) without touching frozen GT v1.0,
Stage J, Stage K, or running video decoding / GPU inference.

Compares:
  - Pre-GAP-C (`A0`): `A0_FRONT_ONLY`, `A0_BACK_ONLY`, `A0_DUAL_LEGACY`, `A0_DUAL_FUSED`
  - Post-GAP-C (`A4`): `A4_FRONT_ONLY`, `A4_BACK_ONLY`, `A4_DUAL_LEGACY`, `A4_DUAL_FUSED`,
    `A4_DUAL_FUSED_SPLIT`, `A4_DUAL_FUSED_MINOBS2`

Outputs under `--out` (`results/phase12_task11/` by default):
  - `task11_fusion_results.json`
  - `task11_ablation_summary.csv`
  - `task11_per_pair_events.csv`
  - `task11_forensic_cases.json`
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Any

REPO = Path(os.environ.get("RESEARCH_HELMET_ROOT", str(Path(__file__).resolve().parents[2])))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import evaluate_gapc_tracking as gct  # noqa: E402
import extract as ex  # noqa: E402
from src.association.cross_camera_association import CrossCameraAssociationEngine  # noqa: E402
from src.association.tracklet_models import (  # noqa: E402
    CrossCameraAssociationBatchResult,
    TimestampQualityRecord,
)
from src.fusion import (  # noqa: E402
    DualCameraFusionEngine,
    FusionConfig,
    ViewRiderEvidence,
    fuse_rider_beliefs,
)
from src.pipeline.dual_camera_pipeline import derive_pair_events  # noqa: E402


STAGEJ_DIR = gct.STAGEJ_DIR
REL_DIR = gct.REL_DIR
DEFAULT_OUT_DIR = REPO / "results" / "phase12_task11"
PAIRS = gct.PAIRS


@dataclass
class _CamOutProxy:
    camera_id: str
    canonical_shift_sec: float
    pipeline: Any


def _tiou_legacy(a: tuple[float | None, float | None], b: tuple[float | None, float | None]) -> float | None:
    if a[0] is None or b[0] is None:
        return None
    assert a[1] is not None and b[1] is not None
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0.0 else 0.0


def _tiou_fixed(a: tuple[float | None, float | None], b: tuple[float | None, float | None]) -> float | None:
    if a[0] is None or b[0] is None:
        return None
    assert a[1] is not None and b[1] is not None
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    if union == 0.0:
        return 1.0 if (abs(a[0] - b[0]) <= 1e-9 and abs(a[1] - b[1]) <= 1e-9) else 0.0
    return inter / union


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def _replay_pipelines_and_associations(
    tracker_level: str,
    streams: dict[tuple[str, str], dict[str, Any]],
) -> tuple[
    dict[tuple[str, str], _CamOutProxy],
    dict[str, CrossCameraAssociationBatchResult],
]:
    engine = CrossCameraAssociationEngine.from_yaml()
    cam_outs: dict[tuple[str, str], _CamOutProxy] = {}
    batches: dict[str, CrossCameraAssociationBatchResult] = {}

    for pair in PAIRS:
        pdir = STAGEJ_DIR / pair
        sync_diag = json.loads((pdir / f"{pair}_sync_diagnostics.json").read_text(encoding="utf-8"))
        offset_sec = float(sync_diag["offset_sec"])
        sj_assoc = json.loads((pdir / f"{pair}_association.json").read_text(encoding="utf-8"))
        tq = TimestampQualityRecord(**sj_assoc["timestamp_quality"])

        for cam in ("front", "back"):
            sd = streams[(pair, cam)]
            pipe = gct._make_pipeline_for_mode(tracker_level, cam, sd["native_wh"])
            for bundle, should_update in sd["bundles"]:
                pipe.step(bundle, should_update_state=should_update)
            cam_outs[(pair, cam)] = _CamOutProxy(
                camera_id=cam,
                canonical_shift_sec=offset_sec if cam == "front" else 0.0,
                pipeline=pipe,
            )

        batches[pair] = engine.associate_pipelines(
            pair_id=pair,
            front_pipeline=cam_outs[(pair, "front")].pipeline,
            back_pipeline=cam_outs[(pair, "back")].pipeline,
            timestamp_quality=tq,
            front_canonical_pts_by_frame=streams[(pair, "front")]["pts_map"],
            back_canonical_pts_by_frame=streams[(pair, "back")]["pts_map"],
            front_frame_dimensions=streams[(pair, "front")]["native_wh"],
            back_frame_dimensions=streams[(pair, "back")]["native_wh"],
        )

    return cam_outs, batches


def _score_strict_schema_events(
    pred_events: list[dict[str, Any]],
    gt: dict[str, Any],
    *,
    use_fixed_tiou: bool = True,
) -> dict[str, Any]:
    gt_ev = {(e["pair"], e["front"] or "", e["back"] or ""): e for e in gt["events"]}
    pr_ev: dict[tuple[str, str, str], dict[str, Any]] = {}
    for e in pred_events:
        k = (e["pair_id"], e.get("front_track_id") or "", e.get("back_track_id") or "")
        # Preserve unique key deterministic first entry if a split mode creates one-sided keys
        if k not in pr_ev:
            pr_ev[k] = e

    tiou_fn = _tiou_fixed if use_fixed_tiou else _tiou_legacy
    matches = 0
    identity_matches = 0
    type_mismatches: list[dict[str, Any]] = []
    used_g: set[tuple[str, str, str]] = set()
    used_p: set[tuple[str, str, str]] = set()

    for k in sorted(set(gt_ev) & set(pr_ev)):
        g = gt_ev[k]
        p = pr_ev[k]
        pw = p.get("violation_window_ref_sec") or [None, None]
        t = tiou_fn((g["w0"], g["w1"]), (pw[0], pw[1]))
        if g["w0"] is not None and (t is None or t < 0.5):
            continue
        used_g.add(k)
        used_p.add(k)
        identity_matches += 1
        if g["type"] == p["violation_event_type"]:
            matches += 1
        else:
            type_mismatches.append(
                {
                    "pair": k[0],
                    "front": k[1],
                    "back": k[2],
                    "gt_type": g["type"],
                    "pred_type": p["violation_event_type"],
                }
            )

    fp_excl = 0
    for k, p in pr_ev.items():
        if k in used_p:
            continue
        pair = k[0]
        if (k[1] and (pair, "front", k[1]) in gt["excluded"]) or (
            k[2] and (pair, "back", k[2]) in gt["excluded"]
        ):
            fp_excl += 1

    res = _prf(matches, len(pred_events) - matches, len(gt["events"]) - matches)
    res.update(
        {
            "pred_total": len(pred_events),
            "gt_total": len(gt["events"]),
            "identity_matches": identity_matches,
            "type_mismatches": type_mismatches,
            "fp_on_excluded_tracks": fp_excl,
        }
    )
    return res


def _is_violation_type(etype: str) -> bool:
    return etype in ("DRIVER_NO_HELMET", "PILLION_NO_HELMET", "BOTH_NO_HELMET")


def _score_physical_passes(
    pred_events: list[dict[str, Any]],
    gt: dict[str, Any],
) -> dict[str, Any]:
    """Fair single-camera vs dual-camera physical motorcycle-pass evaluation.

    A predicted event corresponds to a GT pass g whenever `(p.pair == g.pair)` and
    (`p.front == g.front` or `p.back == g.back`). Each GT pass can be credited at most
    once (deduplicated, so linked dual events never double-count). Predicted events
    touching excluded tracks (`excluded_tracks`) are counted honestly in FP unless
    tracking filtered them out.
    """
    gt_events = gt["events"]
    non_excl_preds: list[dict[str, Any]] = []
    excl_preds: list[dict[str, Any]] = []
    for e in pred_events:
        pair = e["pair_id"]
        f_id = e.get("front_track_id") or ""
        b_id = e.get("back_track_id") or ""
        if (f_id and (pair, "front", f_id) in gt["excluded"]) or (
            b_id and (pair, "back", b_id) in gt["excluded"]
        ):
            excl_preds.append(e)
        else:
            non_excl_preds.append(e)

    # Map track -> matching pred event indices (can share across matched dual event)
    by_track: dict[tuple[str, str, str], list[int]] = {}
    for idx, e in enumerate(non_excl_preds):
        pair = e["pair_id"]
        if e.get("front_track_id"):
            by_track.setdefault((pair, "front", e["front_track_id"]), []).append(idx)
        if e.get("back_track_id"):
            by_track.setdefault((pair, "back", e["back_track_id"]), []).append(idx)

    pass_detected = 0
    pass_type_agree = 0
    pass_decided_total = 0
    pass_decided_agree = 0
    viol_gt_total = 0
    viol_type_agree = 0
    viol_binary_detected = 0
    front_only_viol_caught = 0
    back_only_viol_caught = 0
    dual_viol_caught = 0
    matched_pred_indices: set[int] = set()
    matched_viol_pred_indices: set[int] = set()

    for g in gt_events:
        cands: list[int] = []
        if g["front"]:
            cands.extend(by_track.get((g["pair"], "front", g["front"]), []))
        if g["back"]:
            cands.extend(by_track.get((g["pair"], "back", g["back"]), []))
        # Deduplicate while preserving order
        seen_c: set[int] = set()
        uniq_cands = [i for i in cands if not (i in seen_c or seen_c.add(i))]

        is_decided = g["type"] != "INSUFFICIENT_VISUAL_EVIDENCE"
        is_viol_gt = _is_violation_type(g["type"])
        has_f = bool(g["front"])
        has_b = bool(g["back"])
        if is_decided:
            pass_decided_total += 1
        if is_viol_gt:
            viol_gt_total += 1

        if not uniq_cands:
            continue

        pass_detected += 1
        matched_pred_indices.update(uniq_cands)

        # Evaluate predicted events associated with this physical pass
        exact_hit = any(non_excl_preds[i]["violation_event_type"] == g["type"] for i in uniq_cands)
        if exact_hit:
            pass_type_agree += 1
            if is_decided:
                pass_decided_agree += 1
            if is_viol_gt:
                viol_type_agree += 1

        if is_viol_gt:
            viol_cands = [i for i in uniq_cands if _is_violation_type(non_excl_preds[i]["violation_event_type"])]
            if viol_cands:
                viol_binary_detected += 1
                matched_viol_pred_indices.update(viol_cands)
                if has_f and not has_b:
                    front_only_viol_caught += 1
                elif has_b and not has_f:
                    back_only_viol_caught += 1
                elif has_f and has_b:
                    dual_viol_caught += 1

    # Physical-pass type PRF over all 70 passes (crediting non-excluded + counting extra preds as FP)
    unmatched_preds = len(pred_events) - len(matched_pred_indices)
    pass_prf = _prf(pass_type_agree, len(pred_events) - pass_type_agree, len(gt_events) - pass_type_agree)

    # Violation-citation binary PRF over GT violation passes vs predicted violation events
    pred_viol_total = sum(1 for e in pred_events if _is_violation_type(e["violation_event_type"]))
    viol_binary_prf = _prf(
        viol_binary_detected,
        pred_viol_total - len(matched_viol_pred_indices),
        viol_gt_total - viol_binary_detected,
    )

    return {
        "gt_passes_total": len(gt_events),
        "physical_passes_detected": pass_detected,
        "physical_pass_coverage": round(pass_detected / len(gt_events), 4) if gt_events else 0.0,
        "physical_pass_type_agree": pass_type_agree,
        "physical_pass_prf": pass_prf,
        "gt_decided_passes": pass_decided_total,
        "decided_pass_type_agree": pass_decided_agree,
        "decided_pass_accuracy": round(pass_decided_agree / pass_decided_total, 4) if pass_decided_total else 0.0,
        "gt_violation_passes": viol_gt_total,
        "violation_binary_detected": viol_binary_detected,
        "violation_binary_recall": round(viol_binary_detected / viol_gt_total, 4) if viol_gt_total else 0.0,
        "violation_strict_type_agree": viol_type_agree,
        "violation_binary_prf": viol_binary_prf,
        "violation_sources": {
            "front_only_gt_caught": front_only_viol_caught,
            "back_only_gt_caught": back_only_viol_caught,
            "dual_visible_gt_caught": dual_viol_caught,
        },
        "pred_total": len(pred_events),
        "pred_on_excluded_tracks": len(excl_preds),
        "unmatched_pred_events": unmatched_preds,
    }


def _evaluate_arm(
    arm_name: str,
    *,
    tracker_level: str,
    fusion_mode: str,
    fusion_config: FusionConfig | None,
    cam_outs: dict[tuple[str, str], _CamOutProxy],
    batches: dict[str, CrossCameraAssociationBatchResult],
    gt: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    t0 = time.perf_counter()
    engine = DualCameraFusionEngine(config=fusion_config)
    all_events: list[dict[str, Any]] = []
    all_orphans: list[dict[str, Any]] = []
    fusion_counters = Counter()
    rider_decision_states = Counter()

    for pair in PAIRS:
        f_out = cam_outs[(pair, "front")]
        b_out = cam_outs[(pair, "back")]
        batch = batches[pair]
        if fusion_mode == "legacy":
            evs = derive_pair_events(pair, f_out, b_out, batch, fusion_mode="legacy")
            for ev in evs:
                ev["pair_id"] = pair
            all_events.extend(evs)
        else:
            out = engine.fuse_pair(pair, f_out, b_out, batch, mode=fusion_mode)
            for ev in out.events:
                ev["pair_id"] = pair
                fmeta = ev.get("fusion_metadata") or {}
                if fmeta.get("cross_view_fused"):
                    fusion_counters["cross_view_fused_events"] += 1
                if fmeta.get("quarantined"):
                    fusion_counters["quarantined_events"] += 1
                if fmeta.get("has_evidence_conflict_abstain"):
                    fusion_counters["conflict_abstain_events"] += 1
                for r in ev.get("riders", []):
                    rider_decision_states[r.get("fusion_decision_state", "UNKNOWN")] += 1
                all_events.append(ev)
            for orp in out.orphan_results:
                od = orp.to_dict()
                od["pair_id"] = pair
                all_orphans.append(od)

    strict_fixed = _score_strict_schema_events(all_events, gt, use_fixed_tiou=True)
    strict_legacy = _score_strict_schema_events(all_events, gt, use_fixed_tiou=False)
    physical = _score_physical_passes(all_events, gt)
    wall = round(time.perf_counter() - t0, 3)

    summary = {
        "arm": arm_name,
        "tracker_level": tracker_level,
        "fusion_mode": fusion_mode,
        "wall_sec": wall,
        "strict_schema_fixed_tiou": strict_fixed,
        "strict_schema_legacy_tiou": strict_legacy,
        "physical_pass_eval": physical,
        "fusion_event_counters": dict(sorted(fusion_counters.items())),
        "rider_decision_states": dict(sorted(rider_decision_states.items())),
        "orphan_fusion_summary": {
            "total": len(all_orphans),
            "by_decision": dict(sorted(Counter(o["fusion_decision_state"] for o in all_orphans).items())),
            "linked_decided": sum(
                1 for o in all_orphans if o["association_outcome"] == "LINKED" and o["helmet_state"] in ("HELMET", "NO_HELMET")
            ),
            "quarantined": sum(1 for o in all_orphans if o["quarantined"]),
        },
    }
    return summary, all_events, {"events": all_events, "orphans": all_orphans}


def _run_stagej_k1_veto_audit() -> list[dict[str, Any]]:
    """Verify quantitatively how `src/fusion/` fixes Stage J's 3 real K=1 veto pairs."""
    cases = [
        ("pair_012", "front_moto_0004", "front_moto_0004_rider_s0", "back_moto_0002", "back_moto_0002_rider_s0", "HELMET"),
        ("pair_018", "front_moto_0001", "front_moto_0001_rider_s1", "back_moto_0001", "back_moto_0001_rider_s1", "NO_HELMET"),
        ("pair_024", "front_moto_0001", "front_moto_0001_rider_s1", "back_moto_0001", "back_moto_0001_rider_s1", "NO_HELMET"),
    ]
    results: list[dict[str, Any]] = []
    for pair, fm, fs, bm, bs, expected in cases:
        tf_f = json.loads((STAGEJ_DIR / pair / f"{pair}_front_tracks_final.json").read_text(encoding="utf-8"))
        tf_b = json.loads((STAGEJ_DIR / pair / f"{pair}_back_tracks_final.json").read_text(encoding="utf-8"))

        def _extract_hist(tf: dict[str, Any], mid: str, sub_id: str) -> tuple[int, int, float, float]:
            for mt in tf["moto_tracks"]:
                if mt["moto_track_id"] != mid:
                    continue
                for slot, hist in mt.get("rider_pts_history", {}).items():
                    if f"{mid}_rider_s{slot}" == sub_id:
                        h_c = sum(1 for x in hist if x[3] == "helmet")
                        nh_c = sum(1 for x in hist if x[3] == "no_helmet")
                        h_w = sum(float(x[4]) for x in hist if x[3] == "helmet")
                        nh_w = sum(float(x[4]) for x in hist if x[3] == "no_helmet")
                        return h_c, nh_c, h_w, nh_w
            return 0, 0, 0.0, 0.0

        fh, fnh, fhw, fnhw = _extract_hist(tf_f, fm, fs)
        bh, bnh, bhw, bnhw = _extract_hist(tf_b, bm, bs)
        f_ev = ViewRiderEvidence(
            camera_id="front",
            track_id=fs,
            slot_index=0,
            slot_role="driver",
            helmet_count=fh,
            no_helmet_count=fnh,
            helmet_conf_sum=fhw,
            no_helmet_conf_sum=fnhw,
            track_helmet_state="HELMET" if fh > fnh else ("NO_HELMET" if fnh > fh else "UNCERTAIN"),
        )
        b_ev = ViewRiderEvidence(
            camera_id="back",
            track_id=bs,
            slot_index=0,
            slot_role="driver",
            helmet_count=bh,
            no_helmet_count=bnh,
            helmet_conf_sum=bhw,
            no_helmet_conf_sum=bnhw,
            track_helmet_state="HELMET" if bh > bnh else ("NO_HELMET" if bnh > bh else "UNCERTAIN"),
        )
        fused = fuse_rider_beliefs(f_ev, b_ev, canonical_slot=0, role="driver", association_score=0.90)
        results.append(
            {
                "pair_id": pair,
                "front_subtrack": fs,
                "back_subtrack": bs,
                "front_counts_H_NH": [fh, fnh],
                "back_counts_H_NH": [bh, bnh],
                "legacy_stagej_state": "UNCERTAIN",
                "task11_fused_state": fused.helmet_state,
                "task11_decision": fused.fusion_decision_state.value,
                "fused_p_no_helmet": round(fused.fused_p_no_helmet, 4),
                "contradiction_k": round(fused.contradiction_k, 4),
                "expected_state": expected,
                "resolved_k1_veto": fused.helmet_state == expected,
            }
        )
    return results


def _build_forensic_cases(
    arm_outputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    def _find_event(arm: str, pair: str, front_id: str | None, back_id: str | None) -> list[dict[str, Any]]:
        return [
            e
            for e in arm_outputs[arm]["events"]
            if e["pair_id"] == pair
            and (front_id is None or e.get("front_track_id") == front_id)
            and (back_id is None or e.get("back_track_id") == back_id)
        ]

    # pair_015 false-link conflict check
    p15_leg = _find_event("A4_DUAL_LEGACY", "pair_015", "front_moto_0003", "back_moto_0001")
    p15_fus = _find_event("A4_DUAL_FUSED", "pair_015", "front_moto_0003", "back_moto_0001")

    # pair_012 recovered linked driver
    p12_fus = _find_event("A4_DUAL_FUSED", "pair_012", "front_moto_0004", "back_moto_0002")

    # pair_019 cross-slot RiderAssociation + pillion AMBIGUOUS quarantine
    p19_fus = _find_event("A4_DUAL_FUSED", "pair_019", "front_moto_0002", "back_moto_0002")

    # pair_020 AMBIGUOUS_ABSTAIN quarantine (front_moto_0001 <-> back_moto_0003 bicycle)
    p20_leg = _find_event("A4_DUAL_LEGACY", "pair_020", "front_moto_0001", "back_moto_0003")
    p20_fus = _find_event("A4_DUAL_FUSED", "pair_020", "front_moto_0001", "back_moto_0003")
    p20_split_f = [
        e for e in arm_outputs["A4_DUAL_FUSED_SPLIT"]["events"] if e["pair_id"] == "pair_020" and e.get("front_track_id") == "front_moto_0001"
    ]
    p20_split_b = [
        e for e in arm_outputs["A4_DUAL_FUSED_SPLIT"]["events"] if e["pair_id"] == "pair_020" and e.get("back_track_id") == "back_moto_0003"
    ]

    # Single-camera complementarity showcase (pair_001, pair_005, pair_010, pair_017)
    comp_examples = {}
    for pair in ("pair_001", "pair_005", "pair_010"):
        comp_examples[pair] = {
            "front_only": [e["violation_event_type"] for e in arm_outputs["A4_FRONT_ONLY"]["events"] if e["pair_id"] == pair],
            "back_only": [e["violation_event_type"] for e in arm_outputs["A4_BACK_ONLY"]["events"] if e["pair_id"] == pair],
            "dual_fused": [e["violation_event_type"] for e in arm_outputs["A4_DUAL_FUSED"]["events"] if e["pair_id"] == pair],
        }

    return {
        "stagej_k1_veto_resolutions": _run_stagej_k1_veto_audit(),
        "pair_015_false_link_conflict": {
            "legacy": p15_leg[0] if p15_leg else None,
            "dual_fused": p15_fus[0] if p15_fus else None,
        },
        "pair_012_dual_corroboration": p12_fus[0] if p12_fus else None,
        "pair_019_rider_association_and_quarantine": p19_fus[0] if p19_fus else None,
        "pair_020_bicycle_quarantine": {
            "legacy_merged": p20_leg[0] if p20_leg else None,
            "dual_fused_preserve_keys": p20_fus[0] if p20_fus else None,
            "dual_fused_split_front": p20_split_f[0] if p20_split_f else None,
            "dual_fused_split_back": p20_split_b[0] if p20_split_b else None,
        },
        "single_vs_dual_complementarity_examples": comp_examples,
        "pair_009_orphan_link_fused": [
            o for o in arm_outputs["A4_DUAL_FUSED"]["orphans"] if o["pair_id"] == "pair_009" and o["association_outcome"] == "LINKED"
        ],
    }


def run_all(out_dir: Path, *, write_files: bool = True) -> dict[str, Any]:
    t_total0 = time.perf_counter()
    gt = ex.load_gt()
    streams = gct._preload_all_streams()

    a0_cams, a0_batches = _replay_pipelines_and_associations("A0", streams)
    a4_cams, a4_batches = _replay_pipelines_and_associations("A4", streams)

    specs: list[tuple[str, str, str, FusionConfig | None]] = [
        ("A0_DUAL_LEGACY", "A0", "legacy", None),
        ("A0_FRONT_ONLY", "A0", "front_only", FusionConfig()),
        ("A0_BACK_ONLY", "A0", "back_only", FusionConfig()),
        ("A0_DUAL_FUSED", "A0", "dual_fused", FusionConfig()),
        ("A4_FRONT_ONLY", "A4", "front_only", FusionConfig()),
        ("A4_BACK_ONLY", "A4", "back_only", FusionConfig()),
        ("A4_DUAL_LEGACY", "A4", "legacy", None),
        ("A4_DUAL_FUSED", "A4", "dual_fused", FusionConfig()),
        (
            "A4_DUAL_FUSED_SPLIT",
            "A4",
            "dual_fused",
            FusionConfig(split_ambiguous_moto_to_single_view_events=True),
        ),
        (
            "A4_DUAL_FUSED_MINOBS2",
            "A4",
            "dual_fused",
            FusionConfig(min_observations_for_decision=2),
        ),
    ]

    arm_summaries: dict[str, Any] = {}
    arm_outputs: dict[str, dict[str, Any]] = {}
    per_pair_rows: list[dict[str, Any]] = []

    for arm_name, t_lvl, f_mode, f_cfg in specs:
        cams = a0_cams if t_lvl == "A0" else a4_cams
        bmap = a0_batches if t_lvl == "A0" else a4_batches
        summary, events, payload = _evaluate_arm(
            arm_name,
            tracker_level=t_lvl,
            fusion_mode=f_mode,
            fusion_config=f_cfg,
            cam_outs=cams,
            batches=bmap,
            gt=gt,
        )
        arm_summaries[arm_name] = summary
        arm_outputs[arm_name] = payload

        if arm_name in ("A4_FRONT_ONLY", "A4_BACK_ONLY", "A4_DUAL_LEGACY", "A4_DUAL_FUSED", "A4_DUAL_FUSED_SPLIT"):
            for ev in sorted(
                events,
                key=lambda x: (
                    x["pair_id"],
                    x["global_moto_id"],
                    x.get("front_track_id") or "",
                    x.get("back_track_id") or "",
                ),
            ):
                fmeta = ev.get("fusion_metadata") or {}
                per_pair_rows.append(
                    {
                        "arm": arm_name,
                        "pair_id": ev["pair_id"],
                        "global_moto_id": ev["global_moto_id"],
                        "front_track_id": ev.get("front_track_id") or "",
                        "back_track_id": ev.get("back_track_id") or "",
                        "link_status": ev["cross_camera_link_status"],
                        "association_outcome": ev["association_outcome"],
                        "violation_event_type": ev["violation_event_type"],
                        "event_is_violation": ev["event_is_violation"],
                        "rider_count": ev["rider_count"],
                        "cross_view_fused": fmeta.get("cross_view_fused", False),
                        "quarantined": fmeta.get("quarantined", False),
                        "conflict_abstain": fmeta.get("has_evidence_conflict_abstain", False),
                        "max_k_conflict": fmeta.get("max_contradiction_k", 0.0),
                    }
                )

    forensics = _build_forensic_cases(arm_outputs)

    # Programmatic verification gates
    a4_f = arm_summaries["A4_FRONT_ONLY"]["physical_pass_eval"]
    a4_b = arm_summaries["A4_BACK_ONLY"]["physical_pass_eval"]
    a0_b = arm_summaries["A0_BACK_ONLY"]["physical_pass_eval"]
    a4_dual = arm_summaries["A4_DUAL_FUSED"]["physical_pass_eval"]
    a4_split = arm_summaries["A4_DUAL_FUSED_SPLIT"]["physical_pass_eval"]

    a0_fus = arm_summaries["A0_DUAL_FUSED"]
    a4_fus = arm_summaries["A4_DUAL_FUSED"]
    a4_leg = arm_summaries["A4_DUAL_LEGACY"]

    checks = {
        "c1_gapc_unblocked_back_only_violation_recall_above_a0": (
            a4_b["violation_binary_recall"] > a0_b["violation_binary_recall"]
        ),
        "c2_dual_fused_exceeds_front_only_physical_coverage_and_viol_recall": (
            a4_dual["physical_pass_coverage"] > a4_f["physical_pass_coverage"]
            and a4_dual["violation_binary_recall"] > a4_f["violation_binary_recall"]
        ),
        "c3_dual_fused_exceeds_back_only_physical_coverage_and_viol_recall": (
            a4_dual["physical_pass_coverage"] > a4_b["physical_pass_coverage"]
            and a4_dual["violation_binary_recall"] > a4_b["violation_binary_recall"]
        ),
        "c4_dual_fused_split_exceeds_a4_dual_legacy": (
            arm_summaries["A4_DUAL_FUSED_SPLIT"]["strict_schema_fixed_tiou"]["f1"]
            > a4_leg["strict_schema_fixed_tiou"]["f1"]
            and a4_split["physical_pass_coverage"] > a4_leg["physical_pass_eval"]["physical_pass_coverage"]
            and a4_split["violation_binary_recall"] > a4_leg["physical_pass_eval"]["violation_binary_recall"]
            and a4_split["decided_pass_accuracy"] > a4_leg["physical_pass_eval"]["decided_pass_accuracy"]
        ),
        "c5_all_three_stagej_k1_vetos_resolved": all(
            c["resolved_k1_veto"] for c in forensics["stagej_k1_veto_resolutions"]
        ),
        "c6_pair015_false_link_quarantined_by_conflict_gate": (
            forensics["pair_015_false_link_conflict"]["dual_fused"] is not None
            and forensics["pair_015_false_link_conflict"]["dual_fused"]["fusion_metadata"]["has_evidence_conflict_abstain"]
            and forensics["pair_015_false_link_conflict"]["dual_fused"]["violation_event_type"] == "INSUFFICIENT_VISUAL_EVIDENCE"
        ),
        "c7_pair020_bicycle_quarantined": (
            forensics["pair_020_bicycle_quarantine"]["dual_fused_preserve_keys"] is not None
            and forensics["pair_020_bicycle_quarantine"]["dual_fused_preserve_keys"]["fusion_metadata"]["quarantined"]
            and forensics["pair_020_bicycle_quarantine"]["dual_fused_split_front"]["violation_event_type"] == "DRIVER_NO_HELMET"
            and forensics["pair_020_bicycle_quarantine"]["dual_fused_split_back"]["violation_event_type"] == "INSUFFICIENT_VISUAL_EVIDENCE"
        ),
        "c8_pair019_rider_assoc_cross_slot_linked_and_pillion_quarantined": (
            forensics["pair_019_rider_association_and_quarantine"] is not None
            and forensics["pair_019_rider_association_and_quarantine"]["violation_event_type"] == "BOTH_NO_HELMET"
            and any(
                r["fusion_decision_state"] == "VERIFIED_DUAL_VIOLATION"
                for r in forensics["pair_019_rider_association_and_quarantine"]["riders"]
            )
            and any(
                r["quarantined"] and not r["cross_view_fused"]
                for r in forensics["pair_019_rider_association_and_quarantine"]["riders"]
            )
        ),
        "c9_pair009_orphan_fused_no_helmet": (
            len(forensics["pair_009_orphan_link_fused"]) == 1
            and forensics["pair_009_orphan_link_fused"][0]["helmet_state"] == "NO_HELMET"
            and forensics["pair_009_orphan_link_fused"][0]["implies_confirmed_motorcycle_link"] is False
        ),
        "c10_gapc_cuts_low_confidence_fusion_abstains": (
            a4_fus["rider_decision_states"].get("LOW_CONFIDENCE_ABSTAIN", 0)
            < a0_fus["rider_decision_states"].get("LOW_CONFIDENCE_ABSTAIN", 0)
        ),
    }

    summary_rows: list[dict[str, Any]] = []
    for arm_name, _, _, _ in specs:
        s = arm_summaries[arm_name]
        sf = s["strict_schema_fixed_tiou"]
        ph = s["physical_pass_eval"]
        summary_rows.append(
            {
                "arm": arm_name,
                "tracker_level": s["tracker_level"],
                "fusion_mode": s["fusion_mode"],
                "strict_f1_fixed_tiou": sf["f1"],
                "strict_precision": sf["precision"],
                "strict_recall": sf["recall"],
                "strict_tp": sf["tp"],
                "strict_fp": sf["fp"],
                "strict_fn": sf["fn"],
                "physical_coverage": ph["physical_pass_coverage"],
                "physical_type_f1": ph["physical_pass_prf"]["f1"],
                "decided_pass_acc": ph["decided_pass_accuracy"],
                "viol_binary_recall": ph["violation_binary_recall"],
                "viol_binary_precision": ph["violation_binary_prf"]["precision"],
                "viol_binary_f1": ph["violation_binary_prf"]["f1"],
                "viol_caught_count": ph["violation_binary_detected"],
                "front_only_viol_caught": ph["violation_sources"]["front_only_gt_caught"],
                "back_only_viol_caught": ph["violation_sources"]["back_only_gt_caught"],
                "dual_visible_viol_caught": ph["violation_sources"]["dual_visible_gt_caught"],
                "pred_events_total": ph["pred_total"],
                "conflict_abstains": s["fusion_event_counters"].get("conflict_abstain_events", 0),
                "quarantined_events": s["fusion_event_counters"].get("quarantined_events", 0),
            }
        )

    peak_rss_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0), 2)
    payload = {
        "schema_version": "task11_dual_camera_fusion_v1",
        "pairs_evaluated": len(PAIRS),
        "streams_evaluated": 2 * len(PAIRS),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "arm_summaries": arm_summaries,
        "runtime_metrics": {
            "total_wall_sec": round(time.perf_counter() - t_total0, 3),
            "peak_rss_mb": peak_rss_mb,
        },
    }

    if write_files:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "task11_fusion_results.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (out_dir / "task11_forensic_cases.json").write_text(
            json.dumps(forensics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if summary_rows:
            with open(out_dir / "task11_ablation_summary.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                w.writeheader()
                w.writerows(summary_rows)
        if per_pair_rows:
            with open(out_dir / "task11_per_pair_events.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(per_pair_rows[0].keys()))
                w.writeheader()
                w.writerows(per_pair_rows)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TASK-11 dual-camera fusion & single-vs-dual ablation.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="Output directory.")
    args = parser.parse_args()
    res = run_all(args.out, write_files=True)
    print(f"TASK11_OUT_DIR: {args.out}")
    print(f"ALL_CHECKS_PASSED: {res['all_checks_passed']} ({sum(res['checks'].values())}/{len(res['checks'])})")
    print(f"WALL_SEC: {res['runtime_metrics']['total_wall_sec']} PEAK_RSS_MB: {res['runtime_metrics']['peak_rss_mb']}")
    for k, ok in res["checks"].items():
        print(f"  CHECK {k}: {ok}")
    print("\nABLATION SUMMARY (Physical Pass & Strict GT-70):")
    for arm in (
        "A0_FRONT_ONLY",
        "A0_BACK_ONLY",
        "A0_DUAL_LEGACY",
        "A0_DUAL_FUSED",
        "A4_FRONT_ONLY",
        "A4_BACK_ONLY",
        "A4_DUAL_LEGACY",
        "A4_DUAL_FUSED",
        "A4_DUAL_FUSED_SPLIT",
        "A4_DUAL_FUSED_MINOBS2",
    ):
        s = res["arm_summaries"][arm]
        sf = s["strict_schema_fixed_tiou"]
        ph = s["physical_pass_eval"]
        print(
            f"  {arm:22s} | strictF1={sf['f1']:.4f} (TP={sf['tp']:2d},FP={sf['fp']:2d},FN={sf['fn']:2d})"
            f" | physCov={ph['physical_pass_coverage']:.4f} physF1={ph['physical_pass_prf']['f1']:.4f}"
            f" | violRec={ph['violation_binary_recall']:.4f} ({ph['violation_binary_detected']}/40)"
            f" | decAcc={ph['decided_pass_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
