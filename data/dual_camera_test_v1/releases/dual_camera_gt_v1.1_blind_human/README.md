# `dual_camera_gt_v1.1_blind_human` — Blind Human Annotation Package

**HUMAN VERIFICATION STATUS: `PENDING_HUMAN_REVIEW`**  
- `human_verified`: `false`  
- `executed_human_units`: `0`  
- `pending_human_units`: `92` (`70` baseline physical transits + `22` GAP-C / unlinked candidate transits across all `25` pairs)  
- `fabricated_labels_present`: `false`  

---

## Important Integrity Notice

This directory is the **non-destructive preparation & workqueue release** for the **Blind Dual-Reviewer Human Ground Truth (`v1.1`)** defined in [`docs/phase12/ARGON_TASK12_BLIND_GT_V1_1_PROTOCOL.md`](file:///Users/sayemuddin/Desktop/research_helmet/docs/phase12/ARGON_TASK12_BLIND_GT_V1_1_PROTOCOL.md).

1. **Zero Fabricated Labels:** No AI-generated, LLM-synthesized, or heuristically inferred human verdicts are included. All human response columns (`annotator_a_*`, `annotator_b_*`, `adjudicator_*`) inside [`blind_annotation_workqueue.csv`](file:///Users/sayemuddin/Desktop/research_helmet/data/dual_camera_test_v1/releases/dual_camera_gt_v1.1_blind_human/blind_annotation_workqueue.csv) remain blank with `review_status = PENDING_HUMAN_REVIEW`.
2. **Frozen `v1.0` Preserved:** [`dual_camera_gt_v1.0_ai_assisted_audited`](file:///Users/sayemuddin/Desktop/research_helmet/data/dual_camera_test_v1/releases/dual_camera_gt_v1.0_ai_assisted_audited/README.md) remains 100% immutable (`43/43` release SHA-256 + `39/39` base SHA-256).

---

## Package Contents

- [`release_metadata.json`](file:///Users/sayemuddin/Desktop/research_helmet/data/dual_camera_test_v1/releases/dual_camera_gt_v1.1_blind_human/release_metadata.json) — Explicit status manifest (`PENDING_HUMAN_REVIEW`, corpus counts, workqueue strata, Cohen's $\kappa$ targets).
- [`schema.json`](file:///Users/sayemuddin/Desktop/research_helmet/data/dual_camera_test_v1/releases/dual_camera_gt_v1.1_blind_human/schema.json) — JSON Schema enforcing dual-annotator blinding, Cohen's $\kappa$ tracking, and plate visibility tiers (`PLATE_CLEAR`, `PLATE_PARTIAL`, `PLATE_UNREADABLE_MOTION_BLUR`, `PLATE_NOT_VISIBLE_FRONT_ONLY`).
- [`blind_annotation_workqueue.csv`](file:///Users/sayemuddin/Desktop/research_helmet/data/dual_camera_test_v1/releases/dual_camera_gt_v1.1_blind_human/blind_annotation_workqueue.csv) — Stratified `92`-item blind workqueue (`46` Stratum 1 GAP-C pillion/conflict divergence, `33` Stratum 2 uncertain/ambiguous/delta, `13` Stratum 3 baseline & rear plate audit) with zero AI helmet verdict leakage.
