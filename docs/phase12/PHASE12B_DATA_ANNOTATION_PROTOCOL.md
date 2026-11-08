# PHASE 12B — DUAL-CAMERA BENCHMARK DATA ANNOTATION PROTOCOL

**Project:** An Integrated Dual-Camera Framework for Context-Aware Motorcycle Helmet Violation Detection with Confidence-Aware Temporal Verification  
**Benchmark Dataset:** `data/dual_camera_test_v1/` (25 Synchronized Video Pairs, `pair_001` through `pair_025`)  
**Architect & Lead Engineer:** Gemini 4 Argon  
**Purpose:** Formal methodology, schema specification, and scientific integrity rules for annotating ground-truth events across the 25 synchronized dual-camera pairs prior to Phase 12I quantitative evaluation.

---

## 1. Scientific Integrity Rules for Ground Truth

> [!CAUTION]
> **STRICT PROHIBITION ON MODEL-BOOTSTRAPPED GROUND TRUTH:**
> 1. Do **NOT** use outputs of the system under test (`best.pt`, `yolov8n.pt`, cross-camera matcher, or `EasyOCR`) as unverified ground truth.
> 2. Every motorcycle trajectory, rider helmet status, cross-camera identity link, and license plate transcription in `data/dual_camera_test_v1/annotations/` **must** be verified by human visual inspection across synchronized front and back contact sheets / frame sequences.
> 3. Until human-verified annotations exist in `data/dual_camera_test_v1/annotations/dual_camera_gt_v1.json`, no quantitative cross-camera precision, recall, $F_1$, IDF1, or OCR accuracy number may be claimed.

---

## 2. Annotation Scope Across the 25 Synchronized Video Pairs

The benchmark dataset `data/dual_camera_test_v1/` comprises 25 pairs (`pair_001`–`pair_025`) totaling `208.11 s` of common synchronized overlap:
- **21 Sub-Frame Synchronized Pairs** ($\epsilon_{\text{sync}} \le \pm 10.0\text{ ms}$): `pair_001`–`003`, `006`–`007`, `009`–`014`, `016`–`025`.
- **4 Near-Frame Synchronized Pairs** ($\epsilon_{\text{sync}} \in [\pm 18.0, \pm 38.0]\text{ ms}$): `pair_004`, `pair_005`, `pair_008`, `pair_015`.
- **6 VFR / 30-FPS Back-Camera Pairs**: `pair_013`–`pair_014` (VFR iOS back streams) and `pair_015`–`pair_018` (30 fps iOS back streams paired with 60 fps Android front streams).

Every active motorcycle passing through the roadway within the common overlap window $[t_{\text{start,ref}}, t_{\text{end,ref}}]$ of each pair must be annotated. Parked background motorcycles without riders or distant opposing-lane motorcycles outside the target corridor ($< 20 \times 20\text{ px}$ head size throughout the entire clip) are annotated with `"evaluation_tier": "IGNORE_BACKGROUND"` so false positives and true recalls on the active corridor are measured without ambiguity.

---

## 3. Hierarchical Annotation Entities

Ground truth is structured into four linked levels:

1. **Pair-Level Metadata (`PairAnnotation`)**:
   - `pair_id` (`"pair_001"` .. `"pair_025"`), common overlap window $[t_{\text{start,ref}}, t_{\text{end,ref}}]$ on the Back-anchored reference clock $t_{\text{ref}}$, synchronization tier (`"sub-frame"` vs. `"near-frame"`), and camera geometry (`"inward_facing"` vs. `"co_aligned_pass_through"`).
2. **Global Motorcycle Event (`GlobalMotorcycleEvent`)**:
   - Represents a single real-world physical motorcycle passing through the dual-camera corridor (`global_moto_id`, e.g., `"pair_001_moto_01"`).
   - Links the corresponding Front-camera observation (`front_obs_id: Optional[str]`) and Back-camera observation (`back_obs_id: Optional[str]`), plus an explicit ground-truth cross-camera link status:
     - `"LINKED_BOTH_VIEWS"`: Motorcycle is visible in both $C_F$ and $C_B$ during the overlap window.
     - `"FRONT_VIEW_ONLY"`: Motorcycle is visible only in $C_F$ during the overlap window (e.g., entered/exited before or after overlap in $C_B$).
     - `"BACK_VIEW_ONLY"`: Motorcycle is visible only in $C_B$ during the overlap window.
     - `"AMBIGUOUS_HUMAN_UNCERTAIN"`: Even a human annotator cannot deterministically link the Front and Back observations.
3. **Per-Camera Trajectory Observation (`CameraViewObservation`)**:
   - Records exact entry timestamp (`t_enter_ref_sec`), closest-approach/pass timestamp (`t_peak_ref_sec`), and exit timestamp (`t_exit_ref_sec`) on the reference clock, along with keyframe bounding boxes (`[x1, y1, x2, y2]` for the motorcycle and each rider head at entry, peak scale, and exit) to allow spatial IoU matching against predicted tracks.
   - Records view-level occlusion quality: `"CLEAR"`, `"PARTIAL_OCCLUSION"`, or `"HEAVY_OCCLUSION"`.
4. **Rider-Slot Ground Truth (`RiderGroundTruth`) & License Plate Ground Truth (`PlateGroundTruth`)**:
   - **Per-Rider Slot (`slot = 0` Driver, `slot = 1` Pillion 1, `slot = 2` Pillion 2)**:
     - `gt_helmet_state`: `"helmet"`, `"no_helmet"`, or `"occluded_unknown"`.
     - `headgear_note`: e.g., `"full_face_helmet"`, `"open_face_helmet"`, `"bare_head"`, `"cap_no_helmet"`, `"turban_or_scarf"`, `"hoodie_no_helmet"`.
     - `front_visibility`: `"VISIBLE_CLEAR"`, `"VISIBLE_BLURRED"`, `"OCCLUDED"`, `"NOT_IN_VIEW"`.
     - `back_visibility`: `"VISIBLE_CLEAR"`, `"VISIBLE_BLURRED"`, `"OCCLUDED"`, `"NOT_IN_VIEW"`.
     - `violation_window_ref_sec`: `[t_viol_start_ref, t_viol_end_ref]` (for `no_helmet` riders).
   - **Rear License Plate (`PlateGroundTruth`)**:
     - `plate_visibility_tier`:
       1. `"READABLE_CLEAR"`: Human annotator can read $\ge 75\%$ of the plate characters from the best Back-camera frame.
       2. `"VISIBLE_UNREADABLE_BLUR"`: Plate rectangle is visible on the rear mudguard, but motion blur or resolution ($< 28 \times 12\text{ px}$) prevents human character reading.
       3. `"OCCLUDED"`: Rear mudguard is blocked by another vehicle or frame edge.
       4. `"MISSING_OR_NO_PLATE"`: Physical motorcycle has no rear license plate attached.
     - `gt_plate_text_normalized`: Uppercase alphanumeric/Bengali string (populated if `"READABLE_CLEAR"`, else `null`).
     - `best_plate_frame_back_pts_sec`: Timestamp in $C_B$ where the plate is sharpest, plus `[x1, y1, x2, y2]` plate bounding box.

---

## 4. Formal JSON Schema Specification

All ground-truth annotations will be stored in `data/dual_camera_test_v1/annotations/dual_camera_gt_v1.json` (with a flattened companion summary in `data/dual_camera_test_v1/annotations/dual_camera_gt_events.csv`):

```json
{
  "schema_version": "1.0.0",
  "dataset_id": "dual_camera_test_v1",
  "annotated_by": "human_verified_protocol_phase12h",
  "pairs": [
    {
      "pair_id": "pair_001",
      "common_overlap_start_ref_sec": 0.0,
      "common_overlap_end_ref_sec": 8.059,
      "global_offset_sec": -1.522,
      "sync_uncertainty_ms": 1.0,
      "sync_tier": "sub-frame",
      "global_motorcycle_events": [
        {
          "global_moto_id": "pair_001_moto_01",
          "evaluation_tier": "ACTIVE_CORRIDOR",
          "cross_camera_link_status": "LINKED_BOTH_VIEWS",
          "rider_count": 2,
          "event_is_violation": true,
          "riders": [
            {
              "rider_slot": 0,
              "role": "driver",
              "gt_helmet_state": "helmet",
              "headgear_note": "full_face_helmet",
              "front_visibility": "VISIBLE_CLEAR",
              "back_visibility": "VISIBLE_CLEAR",
              "violation_window_ref_sec": null
            },
            {
              "rider_slot": 1,
              "role": "pillion_1",
              "gt_helmet_state": "no_helmet",
              "headgear_note": "bare_head",
              "front_visibility": "VISIBLE_CLEAR",
              "back_visibility": "VISIBLE_CLEAR",
              "violation_window_ref_sec": [1.80, 4.95]
            }
          ],
          "front_observation": {
            "obs_id": "pair_001_front_m01",
            "t_enter_ref_sec": 1.65,
            "t_peak_ref_sec": 3.10,
            "t_exit_ref_sec": 3.55,
            "occlusion_level": "CLEAR",
            "peak_frame_pts_front_sec": 4.622,
            "peak_moto_bbox_xyxy": [412, 280, 690, 820],
            "peak_rider_head_bboxes_xyxy": [
              {"rider_slot": 0, "bbox": [495, 275, 575, 365]},
              {"rider_slot": 1, "bbox": [555, 255, 620, 335]}
            ]
          },
          "back_observation": {
            "obs_id": "pair_001_back_m01",
            "t_enter_ref_sec": 3.25,
            "t_peak_ref_sec": 3.70,
            "t_exit_ref_sec": 5.40,
            "occlusion_level": "CLEAR",
            "peak_frame_pts_back_sec": 3.700,
            "peak_moto_bbox_xyxy": [380, 310, 640, 860],
            "peak_rider_head_bboxes_xyxy": [
              {"rider_slot": 0, "bbox": [460, 295, 535, 380]},
              {"rider_slot": 1, "bbox": [485, 275, 555, 355]}
            ]
          },
          "plate_ground_truth": {
            "plate_visibility_tier": "READABLE_CLEAR",
            "gt_plate_text_normalized": "DHAKA-METRO-HA-123456",
            "best_plate_frame_back_pts_sec": 3.850,
            "best_plate_bbox_xyxy": [470, 670, 550, 725]
          }
        }
      ]
    }
  ]
}
```

---

## 5. Matching Protocol Between Predicted Tracks and Ground Truth (Phase 12I Evaluator)

To evaluate system outputs against `dual_camera_gt_v1.json` deterministically:
1. **Per-Camera Track-to-Observation Matching**:
   - A predicted `MotoTrack` $T_c$ in camera $c \in \{F, B\}$ is matched to ground-truth `CameraViewObservation` $O_c$ if their temporal overlap IoU on $t_{\text{ref}}$ is $\ge 0.30$ AND the bounding-box spatial IoU at `peak_frame_pts_sec` (or nearest tracked frame within $\pm 0.15\text{ s}$) is $\ge 0.40$.
2. **Rider-Slot Matching**:
   - Within a matched `MotoTrack`, predicted `RiderSubTrack` slots are matched to ground-truth `riders` via head bounding-box IoU ($\ge 0.35$) at the peak frame, with slot-index fallback if one head is momentarily occluded at peak scale.
3. **Cross-Camera Identity Link Evaluation**:
   - Let $(O_F, O_B)$ be a ground-truth `"LINKED_BOTH_VIEWS"` pair. A predicted cross-camera link $(\hat{T}_F, \hat{T}_B)$ is a **True Positive Link (`TP_assoc`)** if $\hat{T}_F$ matches $O_F$, $\hat{T}_B$ matches $O_B$, and `assoc_status == "MATCHED"`.
   - If the system links $\hat{T}_F$ (matching $O_F^{(1)}$) to $\hat{T}_B$ (matching a *different* motorcycle $O_B^{(2)}$), it is penalized as a **Wrong Identity Link (`FP_wrong_link`)**—the most severe error class in multi-camera enforcement.
4. **Stratified OCR Evaluation**:
   - **Plate Localization Recall** is evaluated across all events where `plate_visibility_tier` $\in \{\text{"READABLE\_CLEAR"}, \text{"VISIBLE\_UNREADABLE\_BLUR"}\}$.
   - **OCR Character Error Rate (CER) & Exact/Partial Match** are evaluated on `"READABLE_CLEAR"` events, while `"VISIBLE_UNREADABLE_BLUR"` events test whether the system's quality gate (`PlateQualitySelector` / $\tau_{\text{ocr}}$) properly flags `UNREADABLE_BLUR_OR_OCCLUSION` rather than emitting high-confidence garbage strings.
