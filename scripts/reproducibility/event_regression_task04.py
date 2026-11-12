"""TASK-04 event-level regression on existing Stage J artifacts (no inference).

Re-derives each event's rider helmet states / event type with the NEW canonical
`track_helmet_state` (replayed from rider_pts_history) through the pipeline's own
`_merged_riders` / `_event_type_for`, and compares with the frozen Stage J events.json.
"""
import json, os, sys
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.dirname(__file__))
from src.pipeline.dual_camera_pipeline import _merged_riders, _event_type_for
from track_state_redesign import replay_from_history

SJ = os.path.join(REPO, "results/phase12_stagej")
n = changed = 0
diffs = []
for i in range(1, 26):
    pair = f"pair_{i:03d}"
    evs = json.load(open(f"{SJ}/{pair}/{pair}_events.json"))
    recs = {}
    for cam in ("front", "back"):
        tf = json.load(open(f"{SJ}/{pair}/{pair}_{cam}_tracks_final.json"))
        for mt in tf["moto_tracks"]:
            subs = {}
            rs = mt.get("rider_subtracks") or {}
            for st in (rs.values() if isinstance(rs, dict) else rs):
                h = mt.get("rider_pts_history", {}).get(str(st["slot_index"]), [])
                subs[st["slot_index"]] = {"track_helmet_state": replay_from_history(h)["track_helmet_state"],
                                          "slot_role": st["slot_role"], "alert_ids": []}
            recs[(cam, mt["moto_track_id"])] = {"t_start_ref_sec": 0.0, "t_end_ref_sec": 0.0, "rider_subtracks": subs}
    for e in evs:
        n += 1
        obs = {}
        for cam, k in (("front", "front_track_id"), ("back", "back_track_id")):
            if e.get(k) and (cam, e[k]) in recs:
                obs[cam] = recs[(cam, e[k])]
        riders, detail = _merged_riders(obs)
        etype, viol = _event_type_for(detail)
        old = [(r["rider_slot"], r["helmet_state"], r["helmet_state_provenance"]) for r in e["riders"]]
        new = [(r["rider_slot"], r["helmet_state"], r["helmet_state_provenance"]) for r in riders]
        if old != new or etype != e["violation_event_type"] or viol != e["event_is_violation"]:
            changed += 1
            diffs.append({"pair": pair, "global_moto_id": e["global_moto_id"], "old": [old, e["violation_event_type"], e["event_is_violation"]], "new": [new, etype, viol]})
out = {"events_compared": n, "events_changed": changed, "diffs": diffs,
       "compared_fields": ["rider slot/helmet_state/provenance", "violation_event_type", "event_is_violation"],
       "note": "Windows derive from per-view NO_HELMET sets and track times; alert ids come from online Phase 10 state which is unchanged by design."}
os.makedirs(os.path.join(REPO, "results/phase12_track_state_redesign"), exist_ok=True)
json.dump(out, open(os.path.join(REPO, "results/phase12_track_state_redesign/event_regression.json"), "w"), indent=1)
print(f"events compared={n} changed={changed}")
for d in diffs: print(d)
