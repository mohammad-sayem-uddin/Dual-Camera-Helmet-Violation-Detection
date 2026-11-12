# Phase 12 Integrated Results — `dual_camera_gt_v1.0_ai_assisted_audited` evaluation

Assembled in Stage L (2026-11-09) from the Stage K evaluation artifacts
(`work/stageK/`) and the Stage L forensic audit (`work/stageL/`). Byte-identical
copies; integrity is covered by `checksums.sha256` in this directory (verified
at assembly).

> The reference release is **AI-assisted, audited annotation
> (`AI_ASSISTED_AUDITED`) — not human-verified ground truth.** Stage K's
> observation-level agreement is perfect by construction (the GT derives from
> the same detections the pipeline reproduces); read every number with the
> Performance Report's framing and the Limitations report before citing it.

## Contents

| File | Source | Content |
|---|---|---|
| `metrics_summary.csv` | Stage K | Headline metrics per family with pair-bootstrap 95% CIs and counting bases |
| `stagek_results.json` | Stage K | Full evaluation detail: all families, FP decompositions, confusion matrices, event decomposition, NOT COMPUTABLE declarations |
| `bootstrap_cis.json` | Stage K | Pair-level bootstrap CIs (B = 2000, seed 20261109, unit = pair) |
| `per_pair_stats.json` | Stage K | Per-pair statistics (bootstrap inputs) |
| `sync_by_pair.csv` | Stage K (from Stage J diagnostics) | Per-pair offset / ε_sync / drift / Q5 table for all 25 pairs, incl. labelled ε+\|drift\| bounds |
| `events_detail.csv` | Stage K | Per-event matching detail under the strict pre-registered rule |
| `forensic_audit_results.json` | Stage L | End-to-end integrity re-verification: videos, metadata, code, weights, Pass 1, package, release, headline reconciliation, Stage K trace — verdict CLEAN |

## Headline numbers (all traced to the files above in Stage L)

- Head detection: P 0.689 [0.594, 0.780] / R 1.000 / F1 0.816 [0.745, 0.876]; AP50 0.731
- Motorcycle track level: P 0.820 / R 1.000 / F1 0.901 (all 18 FP tracks = correction-excluded set)
- Rider (6,727 spatially-valid rows): F1 1.000, mean IoU 1.000 (self-consistency; see Limitations)
- Helmet: observation-level perfect on 6,753 decided; track-level agreement 64.1% (134/209)
- Tracking (head basis): IDF1 0.9987 [0.997, 0.9998]; ID switches 0; HOTA not computable (declared)
- Association: moto F1 0.909 [0.737, 1.000], abstention 38.8%; orphan abstention 63.3%
- Events: P 0.753 / R 0.914 / F1 0.826 [0.729, 0.927]; timing errors 0.0 s; all 27 error instances decomposed, residual 0
- Synchronization: pooled median ε_sync 3 ms; near-frame stratum median ε 36 ms, median |drift| 72 ms (max 76 ms) — drift reported, never applied; residuals on drift pairs are lower bounds

Reports: `docs/phase12/INTEGRATED_SYSTEM_{FINAL,PERFORMANCE,SYNCHRONIZATION}_REPORT.md`
and `docs/phase12/INTEGRATED_SYSTEM_RESEARCH_LIMITATIONS.md`.
Evaluation protocol (frozen pre-metrics): `work/stageK/EVAL_PROTOCOL.md`,
sha256 `466f5f316a56536429088f44fe54c05f30161862fda4b8201b616c0e488b9ec2`.
