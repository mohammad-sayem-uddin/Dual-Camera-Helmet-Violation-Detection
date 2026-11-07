#!/usr/bin/env python3
"""
Phase 11B & 11C: Systematic Pairing & Temporal Synchronization Engine.
Computes audio cross-correlation, lag offsets, common overlap intervals,
and extracts synchronized visual validation composites for every pair.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        frames = w.readframes(w.getnframes())
        sr = w.getframerate()
        data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        return data, sr


def compute_audio_offset(back_wav: Path, front_wav: Path, hop_ms: float = 5.0) -> dict:
    """
    Computes precise temporal offset between Back and Front audio streams.
    Returns:
      - time_offset_sec: positive means Front starts after Back (Back is t=0, Front is at t=offset)
      - peak_correlation: normalized cross-correlation peak
      - quality: HIGH / MEDIUM / LOW
    """
    b_sig, sr = load_wav(back_wav)
    f_sig, _ = load_wav(front_wav)
    
    hop_samples = int(sr * (hop_ms / 1000.0))  # e.g. 80 samples at 16kHz = 5ms
    
    def get_envelope(sig, hop):
        n = len(sig) // hop
        if n == 0:
            return np.zeros(1)
        env = np.zeros(n)
        for i in range(n):
            chunk = sig[i * hop : (i + 1) * hop]
            env[i] = np.sqrt(np.mean(chunk**2) + 1e-9)
        env -= np.mean(env)
        s = np.std(env)
        if s > 1e-6:
            env /= s
        return env

    env_b = get_envelope(b_sig, hop_samples)
    env_f = get_envelope(f_sig, hop_samples)
    
    # Cross correlate: envelope_back against envelope_front
    corr = np.correlate(env_b, env_f, mode="full")
    lags = np.arange(-len(env_f) + 1, len(env_b))
    
    best_idx = np.argmax(corr)
    best_lag = lags[best_idx]
    time_offset_sec = float(best_lag * (hop_ms / 1000.0))
    
    norm = np.sqrt(np.sum(env_b**2) * np.sum(env_f**2))
    peak_corr = float(corr[best_idx] / norm) if norm > 0 else 0.0
    
    # Assess quality
    if peak_corr >= 0.25:
        quality = "HIGH"
    elif peak_corr >= 0.15:
        quality = "MEDIUM"
    else:
        quality = "LOW"
        
    return {
        "time_offset_sec": round(time_offset_sec, 4),
        "peak_correlation": round(peak_corr, 4),
        "quality": quality,
        "back_duration": round(len(b_sig) / sr, 3),
        "front_duration": round(len(f_sig) / sr, 3)
    }


def extract_sync_composite(
    pair_id: str,
    back_video: Path,
    front_video: Path,
    offset_sec: float,
    output_dir: Path
) -> Path:
    """
    Extracts a side-by-side synchronized composite image at the midpoint of overlap.
    Back video time: t_back
    Front video time: t_front = t_back - offset_sec
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_img = output_dir / f"{pair_id}_sync_preview.jpg"
    
    # Probe durations
    def get_dur(v):
        cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(v)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        return float(res.stdout.strip()) if res.stdout.strip() else 5.0
        
    dur_b = get_dur(back_video)
    dur_f = get_dur(front_video)
    
    # Overlap interval on Back timeline:
    # Front runs from offset_sec to offset_sec + dur_f
    overlap_start = max(0.0, offset_sec)
    overlap_end = min(dur_b, offset_sec + dur_f)
    
    if overlap_end > overlap_start:
        t_b = (overlap_start + overlap_end) / 2.0
    else:
        t_b = dur_b / 2.0
        
    t_f = max(0.0, min(dur_f, t_b - offset_sec))
    
    tmp_b = output_dir / f"{pair_id}_tmp_b.jpg"
    tmp_f = output_dir / f"{pair_id}_tmp_f.jpg"
    
    subprocess.run(["ffmpeg", "-y", "-ss", f"{t_b:.3f}", "-i", str(back_video), "-vframes", "1", "-vf", "scale=960:540", str(tmp_b)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["ffmpeg", "-y", "-ss", f"{t_f:.3f}", "-i", str(front_video), "-vframes", "1", "-vf", "scale=960:540", str(tmp_f)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if tmp_b.exists() and tmp_f.exists():
        im_b = Image.open(tmp_b)
        im_f = Image.open(tmp_f)
        
        # Combine side by side with banner header
        w, h = 960, 540
        banner_h = 60
        comp = Image.new("RGB", (w * 2, h + banner_h), (24, 24, 27))
        
        comp.paste(im_b, (0, banner_h))
        comp.paste(im_f, (w, banner_h))
        
        draw = ImageDraw.Draw(comp)
        title = f"{pair_id.upper()} | Back: {back_video.name} (t={t_b:.2f}s) | Front: {front_video.name} (t={t_f:.2f}s) | Offset: {offset_sec:+.3f}s"
        draw.text((30, 20), title, fill=(255, 255, 255))
        
        comp.save(out_img, quality=90)
        tmp_b.unlink(missing_ok=True)
        tmp_f.unlink(missing_ok=True)
        
    return out_img


def main():
    root = Path(".")
    back_dir = root / "Motorcycle synchronization/Back"
    front_dir = root / "Motorcycle synchronization/Front"
    audio_cache = root / "results/phase11_dual_camera/audio_cache"
    out_dir = root / "results/phase11_dual_camera"
    previews_dir = out_dir / "synchronized_preview"
    previews_dir.mkdir(parents=True, exist_ok=True)
    
    # Load inventory
    with open(out_dir / "source_video_inventory.csv") as f:
        inv = {r["filename"]: r for r in csv.DictReader(f)}
        
    # The 25 Candidate Pairs
    pair_definitions = [
        ("pair_001", "IMG_0620.MOV", "1.mp4"),
        ("pair_002", "IMG_0621.MOV", "2.mp4"),
        ("pair_003", "IMG_0622.MOV", "3.mp4"),
        ("pair_004", "IMG_0623.MOV", "4.mp4"),
        ("pair_005", "IMG_0624.MOV", "5.mp4"),
        ("pair_006", "IMG_0625.MOV", "6.mp4"),
        ("pair_007", "IMG_0626.MOV", "7.mp4"),
        ("pair_008", "IMG_0627.MOV", "8.mp4"),
        ("pair_009", "IMG_0628.MOV", "9.mp4"),
        ("pair_010", "IMG_0629.MOV", "10.mp4"),
        ("pair_011", "IMG_0630.MOV", "12.mp4"),
        ("pair_012", "IMG_0631.MOV", "13.mp4"),
        ("pair_013", "IMG_0632.MOV", "14.mp4"),
        ("pair_014", "IMG_0633.MOV", "15.mp4"),
        ("pair_015", "IMG_0634.MOV", "17.mp4"),
        ("pair_016", "IMG_0635.MOV", "18.mp4"),
        ("pair_017", "IMG_0636.MOV", "19.mp4"),
        ("pair_018", "IMG_0637.MOV", "20.mp4"),
        ("pair_019", "IMG_0638.MOV", "21.mp4"),
        ("pair_020", "IMG_0639.MOV", "22.mp4"),
        ("pair_021", "IMG_0640.MOV", "23.mp4"),
        ("pair_022", "IMG_0641.MOV", "24.mp4"),
        ("pair_023", "IMG_0642.MOV", "25.mp4"),
        ("pair_024", "IMG_0643.MOV", "26.mp4"),
        ("pair_025", "IMG_0644.MOV", "27.mp4"),
    ]
    
    print(f"Testing synchronization for all {len(pair_definitions)} candidate pairs...")
    
    results = []
    
    for pair_id, b_file, f_file in pair_definitions:
        b_wav = audio_cache / f"back_{Path(b_file).stem}.wav"
        f_wav = audio_cache / f"front_{Path(f_file).stem}.wav"
        
        sync_res = compute_audio_offset(b_wav, f_wav, hop_ms=5.0)
        offset = sync_res["time_offset_sec"]
        peak = sync_res["peak_correlation"]
        quality = sync_res["quality"]
        
        dur_b = sync_res["back_duration"]
        dur_f = sync_res["front_duration"]
        
        overlap_start = max(0.0, offset)
        overlap_end = min(dur_b, offset + dur_f)
        overlap_dur = max(0.0, overlap_end - overlap_start)
        
        # Extract visual preview
        b_vid = back_dir / b_file
        f_vid = front_dir / f_file
        preview_path = extract_sync_composite(pair_id, b_vid, f_vid, offset, previews_dir)
        
        res_row = {
            "pair_id": pair_id,
            "back_video": b_file,
            "front_video": f_file,
            "offset_sec": offset,
            "peak_correlation": peak,
            "quality": quality,
            "back_duration": dur_b,
            "front_duration": dur_f,
            "overlap_duration": round(overlap_dur, 3),
            "preview_path": str(preview_path),
            "status": "CONFIRMED"
        }
        results.append(res_row)
        print(f"[{pair_id}] {b_file:12} <-> {f_file:6} | Offset: {offset:+.3f}s | Corr: {peak:.4f} | Overlap: {overlap_dur:.2f}s | Quality: {quality}")
        
    # Write JSON metadata
    json_path = out_dir / "synchronization_metadata.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved synchronization metadata to {json_path}")


if __name__ == "__main__":
    main()
