#!/usr/bin/env python3
"""
Phase 11: Extract audio for all 52 source videos into a persistent analysis cache.
Standardizes to 16,000 Hz 16-bit mono WAV.
"""
from pathlib import Path
import subprocess
import sys


def extract_audio():
    root = Path(".")
    back_dir = root / "Motorcycle synchronization/Back"
    front_dir = root / "Motorcycle synchronization/Front"
    cache_dir = root / "results/phase11_dual_camera/audio_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    back_files = sorted([f for f in back_dir.iterdir() if f.is_file() and not f.name.startswith(".")])
    front_files = sorted([f for f in front_dir.iterdir() if f.is_file() and not f.name.startswith(".")])
    
    total = len(back_files) + len(front_files)
    count = 0
    
    print(f"Extracting 16kHz mono audio from {len(back_files)} Back and {len(front_files)} Front videos...")
    
    for f in back_files:
        out_wav = cache_dir / f"back_{f.stem}.wav"
        if not out_wav.exists() or out_wav.stat().st_size == 0:
            cmd = ["ffmpeg", "-y", "-i", str(f), "-ac", "1", "-ar", "16000", "-vn", str(out_wav)]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        count += 1
        print(f"[{count}/{total}] Extracted Back: {f.name} -> {out_wav.name}")
        
    for f in front_files:
        out_wav = cache_dir / f"front_{f.stem}.wav"
        if not out_wav.exists() or out_wav.stat().st_size == 0:
            cmd = ["ffmpeg", "-y", "-i", str(f), "-ac", "1", "-ar", "16000", "-vn", str(out_wav)]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        count += 1
        print(f"[{count}/{total}] Extracted Front: {f.name} -> {out_wav.name}")
        
    print(f"\nAll {total} audio streams successfully cached in {cache_dir}.")


if __name__ == "__main__":
    extract_audio()
