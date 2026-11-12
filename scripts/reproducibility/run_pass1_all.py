"""Pass 1 driver: run annotate_pass1.py pair by pair as subprocesses.
Checkpoint = <out>/pair_XXX/pair_XXX_summary.json exists -> skip.
Pairs whose videos are not yet on the VM are skipped and listed at the end.
"""
import json, subprocess, sys, time
from pathlib import Path

import os
REPO = Path(os.environ.get("HELMET_REPO", "/home/hatch/workspace/research_helmet/repo"))
OUT = Path(os.environ.get("HELMET_PASS1_OUT", "/home/hatch/workspace/research_helmet/work/pass1"))
PY = sys.executable
SCRIPT = os.environ.get("HELMET_PASS1_SCRIPT", "/home/hatch/workspace/research_helmet/work/annotate_pass1.py")
LOG = OUT / "pass1_driver.log"
OUT.mkdir(parents=True, exist_ok=True)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def main():
    only = sys.argv[1:] or [f"pair_{i:03d}" for i in range(1, 26)]
    skipped_missing, done, failed = [], [], []
    for pair_id in only:
        summ = OUT / pair_id / f"{pair_id}_summary.json"
        if summ.exists():
            log(f"{pair_id}: already done, skip")
            done.append(pair_id)
            continue
        vf = REPO / "data/dual_camera_test_v1/front" / f"{pair_id}_front.mp4"
        vb = REPO / "data/dual_camera_test_v1/back" / f"{pair_id}_back.mp4"
        if not (vf.exists() and vb.exists()):
            log(f"{pair_id}: videos missing on VM, skip for now")
            skipped_missing.append(pair_id)
            continue
        log(f"{pair_id}: START")
        t0 = time.time()
        r = subprocess.run([PY, SCRIPT, "--pair", pair_id],
                           capture_output=True, text=True)
        (OUT / pair_id).mkdir(parents=True, exist_ok=True)
        (OUT / pair_id / "run_stdout.log").write_text(r.stdout[-20000:])
        (OUT / pair_id / "run_stderr.log").write_text(r.stderr[-20000:])
        if r.returncode == 0 and summ.exists():
            log(f"{pair_id}: DONE in {time.time()-t0:.0f}s")
            done.append(pair_id)
        else:
            log(f"{pair_id}: FAILED rc={r.returncode} (see run_stderr.log)")
            failed.append(pair_id)
    log(f"DRIVER FINISHED done={len(done)} failed={failed} missing={skipped_missing}")


if __name__ == "__main__":
    main()
