#!/usr/bin/env python3
"""
Phase 11A: Source Video Inventory and Metadata Extraction.
Extracts comprehensive metadata, SHA-256 hashes, codecs, durations, and decodability.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def probe_video(filepath: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(filepath)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return {}
    try:
        return json.loads(res.stdout)
    except Exception:
        return {}


def test_decodability(filepath: Path) -> tuple[bool, str]:
    # Test reading 1 frame at start and 1 frame at end
    cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", str(filepath),
        "-vframes", "2",
        "-f", "null",
        "-"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        return True, "DECODE_OK"
    else:
        return False, f"ERROR: {res.stderr.strip()[:100]}"


def extract_metadata(filepath: Path, camera_type: str) -> dict[str, Any]:
    probe_data = probe_video(filepath)
    stat = filepath.stat()
    file_size = stat.st_size
    mtime = stat.st_mtime
    
    sha256_hash = compute_sha256(filepath)
    decodable, decode_status = test_decodability(filepath)
    
    format_info = probe_data.get("format", {})
    streams = probe_data.get("streams", [])
    
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})
    
    # Duration
    duration = float(format_info.get("duration", video_stream.get("duration", 0.0)))
    
    # Resolution
    width = video_stream.get("width", 0)
    height = video_stream.get("height", 0)
    
    # FPS
    r_fps = video_stream.get("r_frame_rate", "0/0")
    avg_fps = video_stream.get("avg_frame_rate", "0/0")
    try:
        num, den = map(float, avg_fps.split("/"))
        fps_val = round(num / den, 3) if den != 0 else 0.0
    except Exception:
        fps_val = 0.0
        
    nb_frames = video_stream.get("nb_frames")
    if not nb_frames or nb_frames == "N/A":
        # Estimate from duration and fps
        nb_frames = int(round(duration * fps_val)) if fps_val > 0 else 0
    else:
        nb_frames = int(nb_frames)
        
    video_codec = video_stream.get("codec_name", "unknown")
    audio_codec = audio_stream.get("codec_name", "none")
    audio_sample_rate = audio_stream.get("sample_rate", "none")
    audio_channels = audio_stream.get("channels", 0)
    
    # Creation timestamp from tags
    tags = format_info.get("tags", {})
    vtags = video_stream.get("tags", {})
    creation_time = (
        tags.get("creation_time") or
        tags.get("com.apple.quicktime.creationdate") or
        vtags.get("creation_time") or
        "unknown"
    )
    
    device_model = (
        tags.get("com.apple.quicktime.model") or
        tags.get("com.android.version") or
        "unknown"
    )
    
    return {
        "filename": filepath.name,
        "camera_type": camera_type,
        "filepath": str(filepath),
        "file_size_bytes": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 2),
        "duration_sec": round(duration, 3),
        "resolution": f"{width}x{height}",
        "width": width,
        "height": height,
        "fps": fps_val,
        "r_frame_rate": r_fps,
        "avg_frame_rate": avg_fps,
        "frame_count": nb_frames,
        "video_codec": video_codec,
        "audio_codec": audio_codec,
        "audio_sample_rate": audio_sample_rate,
        "audio_channels": audio_channels,
        "creation_time": creation_time,
        "mtime": mtime,
        "device_model": device_model,
        "sha256": sha256_hash,
        "decodable": decodable,
        "decode_status": decode_status
    }


def main():
    root = Path(".")
    back_dir = root / "Motorcycle synchronization/Back"
    front_dir = root / "Motorcycle synchronization/Front"
    output_dir = root / "results/phase11_dual_camera"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    back_files = sorted([f for f in back_dir.iterdir() if f.is_file() and not f.name.startswith(".")])
    front_files = sorted([f for f in front_dir.iterdir() if f.is_file() and not f.name.startswith(".")])
    
    print(f"Found {len(back_files)} Back camera videos and {len(front_files)} Front camera videos.")
    
    records = []
    print("\n--- Inspecting Back Camera Videos ---")
    for f in back_files:
        meta = extract_metadata(f, "Back")
        records.append(meta)
        print(f"[Back] {meta['filename']}: {meta['duration_sec']}s, {meta['resolution']}, {meta['fps']}fps, audio={meta['audio_codec']}, sha256={meta['sha256'][:12]}...")
        
    print("\n--- Inspecting Front Camera Videos ---")
    for f in front_files:
        meta = extract_metadata(f, "Front")
        records.append(meta)
        print(f"[Front] {meta['filename']}: {meta['duration_sec']}s, {meta['resolution']}, {meta['fps']}fps, audio={meta['audio_codec']}, sha256={meta['sha256'][:12]}...")
        
    # Write CSV
    csv_path = output_dir / "source_video_inventory.csv"
    fields = [
        "filename", "camera_type", "filepath", "file_size_bytes", "file_size_mb",
        "duration_sec", "resolution", "width", "height", "fps", "r_frame_rate",
        "avg_frame_rate", "frame_count", "video_codec", "audio_codec",
        "audio_sample_rate", "audio_channels", "creation_time", "device_model",
        "sha256", "decodable", "decode_status"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
        
    print(f"\nInventory saved to {csv_path} with {len(records)} entries.")


if __name__ == "__main__":
    main()
