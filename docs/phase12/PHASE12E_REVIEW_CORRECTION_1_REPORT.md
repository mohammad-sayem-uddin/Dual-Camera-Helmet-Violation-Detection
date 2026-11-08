# Phase 12E — Review Correction 1: Candidate Gating and Margin Isolation in Global Assignment

## 1. Executive Summary

This report documents the resolution of the **Phase 12E Review Correction 1** defect in `src/association/global_assignment.py`. 

During researcher review of commit `981ba393d0d10d7e54c6df81a7a9f92119022221`, it was observed that although `gated_mat` was allocated, the global assignment solver constructed `cost_mat = 1.0 - score_mat` using every candidate pair's `composite_score` without checking `passed_candidate_gate`. Consequently, candidates that failed hard temporal, overlap, or search window candidate gates could participate in SciPy's Hungarian assignment optimization, potentially blocking valid associations. In addition, failed-gate candidates improperly influenced competitor lists, second-best scores, uniqueness margins, 2×2 swap margins, and unassigned tracklet ambiguity evaluations.

This defect has been completely resolved through:
1. Strict candidate gating requiring `passed_candidate_gate == True` and `active_cues_count >= min_active_cues_required` before any candidate can enter Hungarian optimization.
2. A mathematically rigorous **augmented partial bipartite matching formulation** using slack/dummy nodes of size \((m + n) \times (m + n)\), guaranteeing that tracklets can remain unmatched at zero penalty rather than forcing invalid candidate pairs.
3. Candidate margin isolation: row competitors, column competitors, and 2×2 swap calculations strictly evaluate over valid candidates only.
4. Retention of full diagnostic provenance and audit rejection reasons for rejected candidates without allowing them to corrupt optimization.
5. Implementation of 5 comprehensive regression tests (Tests A–E) in `tests/test_phase12e_association.py`.

Full repository test suite passed with 81 passed tests (0 failures, 5 non-fatal library deprecation warnings) in 26.17s.

---

## 2. Root Cause Analysis

In the initial implementation of Phase 12E (`src/association/global_assignment.py`):
```python
# PREVIOUS DEFECTIVE CODE:
score_mat = np.zeros((m, n), dtype=np.float64)
gated_mat = np.zeros((m, n), dtype=bool)
for i, ft in enumerate(eligible_front):
    for j, bt in enumerate(eligible_back):
        cs = score_lookup.get((ft.moto_track_id, bt.moto_track_id))
        if cs is not None:
            score_mat[i, j] = float(cs.composite_score)
            gated_mat[i, j] = bool(cs.passed_candidate_gate)

# Gated_mat was NEVER used to mask cost_mat!
cost_mat = 1.0 - score_mat
row_ind, col_ind = linear_sum_assignment(cost_mat)
```

Because `linear_sum_assignment` in SciPy forces a maximum cardinality matching of size \(\min(m, n)\) across all entries in `cost_mat`:
1. **Valid Association Blocking:** If an invalid candidate \((i, j)\) (e.g. transit time was physically infeasible, but spatial/visual cues yielded a high score like \(0.95\)) had a lower cost \(1.0 - 0.95 = 0.05\), Hungarian optimization paired \(i\) with \(j\). Later, post-filtering rejected \((i, j)\) from linking, but the matching had already blocked \(i\) and \(j\) from pairing with their legitimate cross-camera counterparts.
2. **Margin Corruption & False Ambiguity:** In calculating uniqueness margins:
   ```python
   row_competitors = [float(score_mat[i, j2]) for j2 in range(n) if j2 != j]
   col_competitors = [float(score_mat[i2, j]) for i2 in range(m) if i2 != i]
   ```
   Invalid candidates were included in `row_competitors` and `col_competitors`. If a failed-gate candidate had a score close to a valid candidate, it falsely inflated `second_best`, causing `rc_margin < ambiguous_margin_threshold` and triggering a false `AMBIGUOUS_ABSTAIN`.
3. **2×2 Swap Margin Corruption:** 2×2 alternative swap margins evaluated cross-terms even when one or both cross-terms were physically impossible invalid candidates.
4. **Remaining Tracklet Ambiguity Corruption:** Unassigned tracklets evaluated `np.argmax(score_mat[i, :])` against invalid candidates, allowing an invalid candidate to trigger false `AMBIGUOUS_ABSTAIN`.

---

## 3. Corrected Assignment Behavior & Mathematical Partial Matching Design

### 3.1 Candidate Validity Gate
A candidate pair \((i, j)\) is defined as valid if and only if:
```python
is_valid = (
    cs is not None
    and cs.passed_candidate_gate
    and cs.active_cues_count >= self.config.min_active_cues_required
)
```
Invalid candidates have `valid_mask[i, j] = False` and `valid_score_mat[i, j] = 0.0`.

### 3.2 Augmented Cost Matrix for Partial Bipartite Matching
To allow partial bipartite matching where any Front or Back tracklet can remain unmatched without forcing invalid pairings, an augmented square cost matrix of size \(N \times N\) (where \(N = m + n\)) is constructed:
- **Rows \(0 \dots m - 1\):** Real Front tracklets \(i \in [0, m)\).
- **Rows \(m \dots m + n - 1\):** Dummy Front tracklets \(k \in [0, n)\), corresponding to Real Back tracklets.
- **Columns \(0 \dots n - 1\):** Real Back tracklets \(j \in [0, n)\).
- **Columns \(n \dots n + m - 1\):** Dummy Back tracklets \(l \in [0, m)\), corresponding to Real Front tracklets.

The matrix entries are partitioned into 4 blocks:
1. **Block 1 (Top-Left \(m \times n\)): Real Front to Real Back**
   \[
   C[i, j] = 
   \begin{cases}
   1.0 - S_{i, j} \in [0.0, 1.0] & \text{if } \text{valid\_mask}[i, j] = \text{True} \\
   10^6 & \text{if } \text{valid\_mask}[i, j] = \text{False}
   \end{cases}
   \]
2. **Block 2 (Top-Right \(m \times m\)): Real Front to Dummy Back**
   Real Front \(i\) matching Dummy Back \(i\) represents leaving Front \(i\) unmatched:
   \[
   C[i, n + i] = 1.0, \quad C[i, n + l] = 10^6 \quad (l \ne i)
   \]
3. **Block 3 (Bottom-Left \(n \times n\)): Dummy Front to Real Back**
   Real Back \(j\) matching Dummy Front \(j\) represents leaving Back \(j\) unmatched:
   \[
   C[m + j, j] = 1.0, \quad C[m + k, j] = 10^6 \quad (k \ne j)
   \]
4. **Block 4 (Bottom-Right \(n \times m\)): Dummy Front to Dummy Back**
   Zero-cost slack matching:
   \[
   C[m + k, n + l] = 0.0 \quad \forall k \in [0, n), l \in [0, m)
   \]

**Mathematical Invariants:**
- Total cost of leaving all real tracklets unmatched is exactly \(m \times 1.0 + n \times 1.0 = m + n\).
- Pairing a valid candidate \((i, j)\) yields cost \((1.0 - S_{i, j}) + 0.0 = 1.0 - S_{i, j}\). The cost savings relative to leaving both unmatched is \(2.0 - (1.0 - S_{i, j}) = 1.0 + S_{i, j} \ge 1.0 > 0\). Every valid candidate strictly improves the objective function.
- Pairing an invalid candidate incurs cost \(\ge 10^6 \gg m + n\). An invalid candidate is mathematically guaranteed **never** to be chosen by Hungarian assignment.
- Real-to-real matched pairs extracted from `linear_sum_assignment` satisfy \(r < m, c < n\) and `valid_mask[r, c] == True`.

### 3.3 Margin Calculation Isolation
For each matched pair \((i, j)\):
- **Row competitors:** `[valid_score_mat[i, j2] for j2 != j if valid_mask[i, j2]]`.
- **Column competitors:** `[valid_score_mat[i2, j] for i2 != i if valid_mask[i2, j]]`.
- **Second-best score:** `max(row_competitors + col_competitors, default=0.0)`.
- **Uniqueness margin:** `rc_margin = round(best_score - second_best, 6)`.
- **2×2 Swap margin:** Evaluated against any other matched pair \((i2, j2)\) **if and only if both cross-terms are valid**:
  ```python
  if valid_mask[i, j2] and valid_mask[i2, j]:
      diff_2x2 = round((valid_score_mat[i, j] + valid_score_mat[i2, j2]) - 
                       (valid_score_mat[i, j2] + valid_score_mat[i2, j]), 6)
  ```
  If either cross-term failed candidate gates, the 2×2 swap represents an infeasible physical state and cannot participate in swap margin calculations.

### 3.4 Remaining Tracklet Handling
For tracklets not matched to a linked or ambiguous pair:
- Only valid candidates are evaluated for competition against ambiguous tracklets.
- If a tracklet has zero valid candidates:
  - `outcome = UNMATCHED`
  - `association_score = 0.0`
  - `pair_score_detail = cs` (preserving evaluated candidate provenance from `score_lookup`)
  - `decision_reasons = list(cs.gate_rejection_reasons)` (e.g. `TRANSIT_TIME_INFEASIBLE`, `OUTSIDE_OVERLAP_WINDOW`)
  - If `INSUFFICIENT_ACTIVE_CUES` is present without temporal infeasibility, `INSUFFICIENT_EVIDENCE` is preserved.

---

## 4. Verification and Regression Test Suite

Five new regression tests were added to `tests/test_phase12e_association.py`:

| Test Identifier | Description | Result |
| :--- | :--- | :--- |
| **Test A** (`test_review_correction_test_a_...`) | High-scoring failed-gate candidate \((f_1, b_0)\) (\(S=0.96\), failed transit) cannot block valid associations \((f_0, b_0)\) and \((f_1, b_1)\). Strictly 1-to-1 matching preserved. | **PASSED** |
| **Test B** (`test_review_correction_test_b_...`) | Failed-gate candidate \((f_0, b_1)\) (\(S=0.80\)) cannot corrupt margins of valid pair \((f_0, b_0)\) (\(S=0.82\)). Valid pair correctly links; false `AMBIGUOUS_ABSTAIN` prevented; rejection reason preserved on \(b_1\). | **PASSED** |
| **Test C** (`test_review_correction_test_c_...`) | When all 4 candidate pairs fail hard gates, no links are made, all tracklets receive `UNMATCHED`, no global IDs are created, and association scores are 0.0. | **PASSED** |
| **Test D** (`test_review_correction_test_d_...`) | Partial matching across differing camera counts (\(3\) Front, \(2\) Back) links legitimate pairs and leaves remaining tracklet `UNMATCHED`. Competing candidates for a single target correctly resolve to `AMBIGUOUS_ABSTAIN`. | **PASSED** |
| **Test E** (`test_review_correction_test_e_...`) | Hand-constructed matrix verifies exact row, column, and 2×2 swap margins. Invalid cross-terms are excluded from 2×2 swaps; all-valid matrix produces exact hand-calculated swap difference (\(0.35\)). | **PASSED** |

### Test Execution Summary
- **Phase 12E Test Suite (`pytest tests/test_phase12e_association.py -v`):**
  - **31 passed in 3.63s** (26 previous tests + 5 new regression tests, 0 failures).
- **Full Repository Test Suite (`/opt/homebrew/bin/python3.11 -m pytest tests/ -v`):**
  - **81 passed, 5 warnings in 26.17s** (0 failures).

---

## 5. Scope Preservation Confirmation

1. **Original Video Files:** Untouched and unmodified.
2. **Phase 11.6 Synchronization Metadata:** Untouched and unmodified.
3. **Phase 12C Synchronization Reader:** Untouched and unmodified.
4. **Phase 10 Temporal State Verification:** Untouched and strictly independent.
5. **Phase 12D Rider Identity Protection:** Untouched and preserved.
6. **Model Training / Fine-Tuning:** No models were trained or fine-tuned.
7. **Next Phases:** Phase 12F and Phase 12G have **NOT** been started.

---

## 6. Scientific Limitations

1. **Uncalibrated Transit Thresholds:** The temporal search window and velocity boundaries remain operational engineering hypotheses pending multi-crossing field calibration.
2. **Missing Ground Truth:** Cross-camera association metrics are validated using synthetic fixtures and deterministic multi-camera invariants; true positive/false positive rates on real unstructured traffic cannot be certified without human-annotated cross-camera ground truth.
3. **Conservative Posture:** By strictly excluding failed candidates and maintaining ambiguity margins, the system errs on the side of abstention (`UNMATCHED` / `AMBIGUOUS_ABSTAIN`) rather than risking false cross-camera identity links.
