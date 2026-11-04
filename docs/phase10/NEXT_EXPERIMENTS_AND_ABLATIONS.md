# Future Experiments and Ablation Study Plan

**Project:** An Integrated Dual-Camera Framework for Context-Aware Motorcycle Helmet Violation Detection with Confidence-Aware Temporal Verification  
**Phase:** 10 Post-Implementation Planning  
**Status:** Theoretical Specification & Experimental Protocols (Do not execute without authorization)  

---

## 1. Motivation

While Phase 10 established the core temporal verification architecture and proved 100% false-positive noise burst suppression in controlled synthetic tests, real-world deployment across dense metropolitan traffic demands rigorous ablation studies.

This document outlines seven controlled ablation experiments designed to optimize hyperparameter sensitivity, assess edge-case robustness, and prepare for multi-camera integration.

---

## 2. Planned Ablation Studies

### Ablation 1: Temporal Window Length ($W$) Sensitivity
- **Objective:** Determine the Pareto frontier between confirmation latency and false alert suppression.
- **Hypothesis:** Smaller windows ($W \le 5$) minimize latency ($< 150\text{ ms}$) but are vulnerable to multi-frame noise bursts. Larger windows ($W \ge 20$) eliminate noise but delay citations beyond camera exit zones.
- **Protocol:**
  - Test values: $W \in \{3, 5, 8, 10, 15, 20, 30\}$ frames.
  - Evaluate on synthetic noise bursts of lengths $k \in \{1, 2, 3, 5\}$ frames.
  - Record: Confirmation latency (ms), false alert rate (%), and abstention rate (%).

### Ablation 2: Aggregator Comparison Under Controlled Detection Noise
- **Objective:** Quantitatively compare Simple Voting, Confidence-Weighted Aggregation, and EMA Smoothing under simulated detector degradation.
- **Hypothesis:** Confidence-weighted aggregation achieves lower false-positive rates than simple voting when false detections have lower confidence scores ($c \in [0.25, 0.45]$) than true detections ($c \in [0.80, 0.95]$).
- **Protocol:**
  - Inject Gaussian noise into detection confidence: $c' = \text{clip}(c + \mathcal{N}(0, \sigma^2), 0, 1)$ for $\sigma \in \{0.05, 0.10, 0.20, 0.30\}$.
  - Inject random class-flip probabilities: $p_{flip} \in \{0.02, 0.05, 0.10, 0.15\}$.
  - Measure Area Under the Precision-Recall Curve (AUC-PR) at event level.

### Ablation 3: Dual-Threshold Hysteresis Gap ($\Delta_{hyst}$)
- **Objective:** Evaluate the impact of the hysteresis separation gap $\Delta = \tau_{confirm} - \tau_{clear}$ on state stability.
- **Hypothesis:** Zero hysteresis ($\tau_{confirm} = \tau_{clear} = 0.50$) induces rapid state flickering at decision boundaries, whereas a separation gap $\Delta \ge 0.25$ ensures monotonic state transitions.
- **Protocol:**
  - Hold $\tau_{confirm} = 0.70$ constant; sweep $\tau_{clear} \in \{0.20, 0.30, 0.40, 0.50, 0.60, 0.70\}$.
  - Quantify State Stability Score and state transition frequency on oscillating trajectories.

### Ablation 4: Occlusion Grace Period ($N_{grace}$) vs. Identity Fragmentation
- **Objective:** Balance track preservation across occlusions against ghost track proliferation.
- **Hypothesis:** Setting $N_{grace} = 5$ frames ($\approx 167\text{ ms}$) recovers $> 95\%$ of urban roadside occlusions without associating unrelated downstream riders.
- **Protocol:**
  - Sweep $N_{grace} \in \{2, 4, 6, 8, 10, 15\}$ frames.
  - Measure Identity Switch Rate (IDSW) and Multiple Object Tracking Accuracy (MOTA) on multi-rider crossing scenes.

### Ablation 5: Visual Appearance Embeddings (ReID) vs. Spatial IoU Tracking
- **Objective:** Assess tracking degradation in dense traffic clusters where 2D bounding boxes overlap significantly.
- **Hypothesis:** In traffic densities $> 4$ motorcycles per lane, adding a lightweight visual appearance embedding (e.g. MobileNetV3 / OSNet) reduces identity switches by $> 40\%$ compared to pure IoU matching.
- **Protocol:**
  - Benchmark ByteTrack IoU association against Hybrid IoU + Cosine Appearance distance.
  - Measure inference latency penalty on edge hardware (Jetson / Raspberry Pi / CPU).

### Ablation 6: Temporal Verification vs. Single-Frame Decision in Roadside Deployment
- **Objective:** Quantify real-world citation precision improvement over baseline single-frame detection.
- **Hypothesis:** Downstream temporal verification reduces roadside false violation citations by $> 85\%$ while maintaining overall violation recall within $2\%$ of single-frame detection.
- **Protocol:**
  - Compare raw frame predictions from `best.pt` directly triggering citations versus the complete temporal state machine pipeline on continuous sequences.

### Ablation 7: Dual-Camera Temporal Homography and Spatial Fusion (Phase 11 Gateway)
- **Objective:** Extend single-stream temporal verification to synchronized front-angle and overhead CCTV streams.
- **Protocol:**
  - Compute homography matrix $\mathbf{H}_{1 \to 2}$ between camera viewpoints.
  - Map track centroid trajectories: $\mathbf{x}_2 = \mathbf{H} \mathbf{x}_1$.
  - Combine multi-view temporal confidence: $\bar{C}_{joint}(t) = \beta C_{cam1}(t) + (1 - \beta) C_{cam2}(t)$.

---

## 3. Execution Constraints
These ablations represent future experimental directions. They are documented here to provide a clear academic research roadmap and must not be launched until explicitly authorized.
