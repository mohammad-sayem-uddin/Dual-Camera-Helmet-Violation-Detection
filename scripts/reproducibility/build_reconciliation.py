#!/usr/bin/env python3
"""Task 3 Step 4 — build frame_reconciliation.csv (50 rows).

Columns per the task spec:
  filename, phase12c_decoded_claimed, phase12c_packet_or_forensic,
  measured_packet_count, measured_nb_frames, measured_sequential_decode,
  stagej_processed, stagej_coverage, excluded_frames, difference, explanation

phase12c_decoded_claimed = opencv_decoded_frame_count in the 12C forensic CSV.
phase12c_packet_or_forensic = frame_count (ffprobe packet-PTS list length) in the same CSV.
stagej_processed / stagej_coverage = Stage J pair_summary coverage (decoded == coverage rows;
Pass 1 verified identical per pair and its coverage CSVs carry one row per decoded frame).
The 15,637 figure appears ONLY as prose totals in the two 12C reports; the 12C phase produced
no per-video numbers that sum to it, so it cannot be distributed per video — rows therefore
carry the phase's actual per-video artifact values, and the prose discrepancy is handled in
the aggregate accounting inside TASK3_FRAME_RECONCILIATION.md.
"""
import csv
import json
import os

from pathlib import Path

REPO = Path(os.environ.get("RESEARCH_HELMET_ROOT", str(Path(__file__).resolve().parents[2])))
PC = os.environ.get("POSTCOMPLETION_OUT_DIR", str(REPO / "results" / "postcompletion"))


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    forensic = {r["filename"]: r for r in load_csv(str(
        REPO / "results/phase12c_pts_sync/stream_pts_forensics_50_videos.csv"))}
    measured = {r["filename"]: r for r in load_csv(os.path.join(PC, "frame_counts_measured.csv"))}
    stagej = json.load(open(os.path.join(PC, "stagej_per_pair.json")))   # pair -> [front, back]
    pass1 = json.load(open(os.path.join(PC, "pass1_per_pair.json")))
    assert stagej == {k: list(v) for k, v in pass1.items()}, "Pass1 != StageJ per pair"

    rows = []
    for fn in sorted(forensic):
        pair = fn.rsplit("_", 1)[0]
        cam = fn.rsplit("_", 1)[1].replace(".mp4", "")
        idx = 0 if cam == "front" else 1
        f12 = forensic[fn]
        m = measured[fn]
        sj = int(stagej[pair][idx])
        claimed_dec = int(f12["opencv_decoded_frame_count"])
        claimed_pkt = int(f12["frame_count"])
        meas_pkt = int(m["ffprobe_packet_count"])
        meas_nb = int(m["ffprobe_nb_frames"])
        meas_dec = int(m["opencv_sequential_decode"])
        excluded = meas_dec - sj
        diff = meas_pkt - sj
        if claimed_dec == claimed_pkt == meas_pkt == meas_nb == meas_dec == sj:
            expl = ("All six counts identical. The 15,637 figure exists only as a prose total in the "
                    "two Phase 12C reports and is not supported by any per-video artifact; this video "
                    "contributes 0 to the 396-frame prose discrepancy.")
        else:
            expl = (f"COUNTS DIFFER: 12C artifact decoded={claimed_dec} packet={claimed_pkt}; measured "
                    f"packet={meas_pkt} nb_frames={meas_nb} decode={meas_dec}; StageJ={sj}.")
        rows.append({
            "filename": fn,
            "phase12c_decoded_claimed": claimed_dec,
            "phase12c_packet_or_forensic": claimed_pkt,
            "measured_packet_count": meas_pkt,
            "measured_nb_frames": meas_nb,
            "measured_sequential_decode": meas_dec,
            "stagej_processed": sj,
            "stagej_coverage": sj,
            "excluded_frames": excluded,
            "difference": diff,
            "explanation": expl,
        })

    out = os.path.join(PC, "frame_reconciliation.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    tot = {k: sum(r[k] for r in rows) for k in rows[0] if k not in ("filename", "explanation")}
    print("rows:", len(rows))
    print("totals:", json.dumps(tot, indent=1))
    print("rows with any mismatch:", [r["filename"] for r in rows
                                      if len({r[c] for c in tot}) > 1])
    print("WROTE", out)


if __name__ == "__main__":
    main()
