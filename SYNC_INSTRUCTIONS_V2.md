# Helmet post-completion sync package v2 (2026-11-12)

Supersedes `helmet_mac_sync_2026-11-09.zip`. Contents (528 files): the complete
mission deliverables — annotation package (40), frozen release
`dual_camera_gt_v1.0_ai_assisted_audited` (44), all `docs/phase12/` reports
including the post-completion `HELMET_TRACK_LEVEL_FAILURE_ANALYSIS.md`,
`HUMAN_VALIDATION_PROTOCOL.md`, `PHASE12_FRAME_COUNT_RECONCILIATION.md` and
`track_failure_examples/` (18 PNGs), `results/phase12_integrated/` (9),
`results/phase12_stagej/` (full 25-pair Stage J inference outputs + run
manifest/config/logs + validation results), `results/phase12_postcompletion/`
(frame-count reconciliation CSV), `src/pipeline/` (2), Stage H tests (3), and
`scripts/reproducibility/` (10). Videos and model weights are deliberately
excluded (already on the Mac, gitignored). `PACKAGE_MANIFEST.sha256` at the
zip root lists the SHA-256 of every file in this package.

## Mac terminal sequence

```bash
cd /Users/sayemuddin/Desktop/research_helmet
unzip -o helmet_mac_sync_postcompletion_2026-11-12.zip
# 1) frozen release checksums — expect the count 43
cd data/dual_camera_test_v1/releases/dual_camera_gt_v1.0_ai_assisted_audited && shasum -c checksums/release_checksums.sha256 | grep -c ': OK' && cd ../../../..
# 2) annotation package checksums — expect the count 39
cd data/dual_camera_test_v1/annotations && shasum -c checksums/annotation_checksums.sha256 | grep -c ': OK' && cd ../../..
# 3) integrated results checksums — expect the count 8
cd results/phase12_integrated && shasum -c checksums.sha256 | grep -c ': OK' && cd ../..
# 4) model weights — expect 0929a882… (helmet) and f59b3d83… (yolov8n)
shasum -a 256 results/phase9_multiseed/exp_combined/seed_42/weights/best.pt yolov8n.pt
# 5) Mac test run (pipeline suites; note any environment failures verbatim)
python3 -m pytest tests/test_phase12_pipeline_synthetic.py tests/test_phase12_pipeline_semantics.py tests/test_phase12_pipeline_real_video.py -q 2>&1 | tail -5
# 6) inspect, then commit and push (never force-push)
git status --short
git diff --stat
git add data/dual_camera_test_v1/annotations data/dual_camera_test_v1/releases docs/phase12 results/phase12_integrated results/phase12_stagej results/phase12_postcompletion scripts/reproducibility src/pipeline tests/test_phase12_pipeline_real_video.py tests/test_phase12_pipeline_semantics.py tests/test_phase12_pipeline_synthetic.py SYNC_INSTRUCTIONS_V2.md PACKAGE_MANIFEST.sha256
git status --short
git commit -m "Helmet: annotation release dual_camera_gt_v1.0_ai_assisted_audited + pipeline, tests, Stage A-L reports, Stage J outputs, post-completion audits"
git push
git log --oneline -1
```

Stop and report back instead of committing if: any checksum count differs
(43 / 39 / 8), `git status` shows anything outside the paths above (especially
any video file or credential), or the push is rejected.
