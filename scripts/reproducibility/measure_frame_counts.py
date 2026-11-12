#!/usr/bin/env python3
"""Task 3 Step 3 — independent per-video frame measurement.

For each of the 50 dataset videos compute:
  (a) ffprobe video-stream packet count  (-count_packets -> nb_read_packets)
  (b) ffprobe stream nb_frames metadata value
  (c) OpenCV sequential decode count (cap.grab() loop, no seeking, no retrieve)
Writes work/postcompletion/frame_counts_measured.csv. Decode-only, no inference.
"""
import csv
import glob
import json
import os
import subprocess
import sys

import cv2

ROOT = os.path.expanduser("~/workspace/research_helmet")
OUT = os.path.join(ROOT, "work/postcompletion/frame_counts_measured.csv")


def ffprobe_counts(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets,nb_frames,codec_name",
         "-of", "json", path],
        capture_output=True, text=True, check=True)
    st = json.loads(out.stdout)["streams"][0]
    return int(st["nb_read_packets"]), int(st["nb_frames"]), st.get("codec_name", "")


def opencv_decode_count(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return -1, -1
    prop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = 0
    while True:
        if not cap.grab():
            break
        n += 1
    cap.release()
    return n, prop


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "repo/data/dual_camera_test_v1/front/*.mp4")) +
                   glob.glob(os.path.join(ROOT, "repo/data/dual_camera_test_v1/back/*.mp4")))
    assert len(files) == 50, len(files)
    rows = []
    for i, path in enumerate(files, 1):
        fn = os.path.basename(path)
        cam = "front" if "/front/" in path else "back"
        pkts, nb_frames, codec = ffprobe_counts(path)
        dec, prop = opencv_decode_count(path)
        rows.append({"filename": fn, "camera": cam, "codec": codec,
                     "ffprobe_packet_count": pkts, "ffprobe_nb_frames": nb_frames,
                     "opencv_sequential_decode": dec, "opencv_cap_prop_frame_count": prop})
        print(f"[{i}/50] {fn}: packets={pkts} nb_frames={nb_frames} decode={dec} prop={prop}",
              flush=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for key in ("ffprobe_packet_count", "ffprobe_nb_frames", "opencv_sequential_decode",
                "opencv_cap_prop_frame_count"):
        print(f"TOTAL {key} =", sum(r[key] for r in rows))
    print("WROTE", OUT)


if __name__ == "__main__":
    sys.exit(main())
