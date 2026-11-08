# Phase 12E — Review Correction 3: End-to-End Consistent Threshold Policy and Rounding-Boundary Elimination

## 1. Executive Summary

This report documents the resolution of the **Phase 12E Review Correction 3** defect in `src/association/global_assignment.py`.

During researcher review following Review Correction 2, a rounding-boundary inconsistency defect was identified:
- Candidate optimization eligibility used a 6-decimal-place rounded comparison:
  ```python
  round(float(cs.composite_score), 6) >= round(self.config.min_association_score, 6)
  ```
- Post-assignment validation used a raw float comparison:
  ```python
  if best_score < self.config.min_association_score:
      continue
  ```
For any score infinitesimally below threshold that rounded up (e.g. \(S = 0.6199996\) with threshold \(\tau = 0.62\)), the candidate was admitted to Hungarian optimization (\(0.620000 \ge 0.620000\)), consumed a tracklet, and was subsequently discarded by raw post-validation (\(0.6199996 < 0.62\)). This recreated the original blocking defect at the rounding boundary.

This defect has been completely eliminated through:
1. Adopting a single, uniform, explicit threshold policy end-to-end: **raw, unrounded, inclusive float comparison** (`float(cs.composite_score) >= float(self.config.min_association_score)`).
2. Applying this exact policy consistently to candidate optimization eligibility, assignment validation, LINKED decision logic, and remaining tracklet classification.
3. Guaranteeing that no candidate can enter Hungarian optimization and subsequently be discarded by post-validation.
4. Clarifying Test F reporting: the test demonstrates that sub-threshold candidates cannot block assignment selection (enabling the higher-cardinality 2-pair matching), and correctly receive `AMBIGUOUS_ABSTAIN` under conservative uniqueness-margin safeguards due to the competing alternative candidate scored `0.99`.
5. Adding four new regression tests (Tests J, K, L, M) in `tests/test_phase12e_association.py`.

Full repository test suite passed with 89 passed tests (0 failures, 5 non-fatal library deprecation warnings).

---

## 2. Root Cause Analysis & Defect Mechanism

In Review Correction 2, `round(..., 6)` was introduced in eligibility:
```python
# DEFECTIVE BOUNDARY MECHANISM:
cs.composite_score = 0.6199996
self.config.min_association_score = 0.62

# 1. Eligibility evaluation:
round(0.6199996, 6) -> 0.620000 >= 0.620000 -> True (Admitted to Hungarian matrix)

# 2. Post-assignment evaluation:
0.6199996 < 0.62 -> True -> Discarded via 'continue'
```

Under this boundary condition:
- The sub-threshold candidate \(0.6199996\) was admitted to Hungarian optimization with cost \(1.0 - 0.6199996 = 0.3800004\).
- It occupied a real tracklet, displacing legitimate pairings.
- Post-Hungarian validation discarded the assigned pair because its unrounded score was strictly less than \(0.62\).
- The tracklet was left `UNMATCHED`, recreating the exact blocking failure mode at the boundary \((\tau - 5 \times 10^{-7}, \tau)\).

---

## 3. End-to-End Consistent Threshold Policy

### 3.1 Unified Raw Comparison
The solver now enforces raw, unrounded, inclusive comparison across the entire pipeline:
\[
\text{Eligible}(i, j) \iff \text{float}(\text{cs.composite\_score}) \ge \text{float}(\text{self.config.min\_association\_score})
\]

### 3.2 Invariant Guarantee
Because `valid_mask[i, j]` strictly requires `float(cs.composite_score) >= float(self.config.min_association_score)`, every matched pair \((i, j)\) selected from Hungarian optimization satisfies:
\[
\text{best\_score} \ge \text{self.config.min\_association\_score}
\]
Post-Hungarian discard due to threshold mismatch is mathematically impossible:
```python
# Invariant: best_score < min_association_score is strictly False for all matched pairs
if best_score < self.config.min_association_score:
    continue
```

### 3.3 Boundary Semantics
With threshold \(\tau = 0.62\):
- **Below threshold (\(S = 0.6199996 < 0.62\)):** Ineligible. Cannot enter optimization, cannot block other pairs, produces `UNMATCHED`.
- **Below threshold (\(S = 0.619999 < 0.62\)):** Ineligible. Produces `UNMATCHED`.
- **Exact threshold (\(S = 0.620000 == 0.62\)):** Eligible under inclusive \(\ge\). Enters optimization, links when uniqueness margins permit.
- **Above threshold (\(S = 0.620001 > 0.62\)):** Eligible. Enters optimization, links when uniqueness margins permit.

---

## 4. Test F Reporting Clarification

Test F evaluates the benchmark matrix:

| | B1 | B2 |
|---|---:|---:|
| **F1** | 0.61 | 0.62 |
| **F2** | 0.62 | 0.99 |

Threshold \(\tau = 0.62\), ambiguous margin threshold \(\delta = 0.10\).

### Exact Behavior:
1. **Selection:**
   - Candidate \(F1 \to B1\) has score \(0.61 < 0.62\). It is excluded from Hungarian optimization.
   - Hungarian optimization evaluates feasible candidates and chooses \(\{ (F1, B2), (F2, B1) \}\) (cost \(0.38 + 0.38 = 0.76\)) instead of consuming \(F1\) on \(B1\).
   - Thus, \(F1 \to B1\) **does not block** \(F1 \to B2\), and \(F2 \to B2\) does not prevent the 2-pair feasible assignment.
2. **Ambiguity Outcome:**
   - For \((F1, B2)\) (\(S = 0.62\)): column competitor in \(B2\) is \(F2 \to B2\) with score \(0.99\). Second best = \(0.99\). Uniqueness margin = \(0.62 - 0.99 = -0.37 < 0.10\).
   - For \((F2, B1)\) (\(S = 0.62\)): row competitor in \(F2\) is \(F2 \to B2\) with score \(0.99\). Second best = \(0.99\). Uniqueness margin = \(0.62 - 0.99 = -0.37 < 0.10\).
   - Both pairs receive `AMBIGUOUS_ABSTAIN`.
3. **Clarification:**
   The test demonstrates that sub-threshold candidates cannot block assignment selection. It does **not** produce two confirmed `LINKED` outcomes because the strong alternative (\(0.99\)) correctly triggers conservative ambiguity abstention. Weakening ambiguity thresholds to force `LINKED` outcomes on Test F would compromise safety.

---

## 5. Regression Test Results (Tests J–M)

Four new regression tests were added to `tests/test_phase12e_association.py`:

| Test Identifier | Purpose | Result |
| :--- | :--- | :--- |
| **Test J** (`test_review_correction_test_j_rounding_boundary_blocking`) | Score `0.6199996` with threshold `0.62`: must not enter Hungarian optimization or block `F1->B2`. Pairs \((F1, B2)\) and \((F2, B1)\) selected. Zero post-optimization discards. | **PASSED** |
| **Test K** (`test_review_correction_test_k_just_below_threshold`) | Score `0.619999 < 0.62`: remains ineligible, produces `UNMATCHED` with `BELOW_MIN` provenance. | **PASSED** |
| **Test L** (`test_review_correction_test_l_exact_threshold`) | Score `0.620000 == 0.62`: remains eligible under inclusive policy, produces `LINKED` with global ID. | **PASSED** |
| **Test M** (`test_review_correction_test_m_just_above_threshold`) | Score `0.620001 > 0.62`: remains eligible, produces `LINKED` with global ID. | **PASSED** |

### Test Suite Execution
- **Phase 12E Test Suite (`pytest tests/test_phase12e_association.py -v`):**
  - **39 / 39 passed** in 3.89s (0 failures).
- **Full Repository Test Suite (`/opt/homebrew/bin/python3.11 -m pytest tests/ -v`):**
  - **89 / 89 passed**, 5 library deprecation warnings, 0 failures.

---

## 6. Scope Preservation Confirmation
- Original video files: **Untouched**.
- Phase 11.6 synchronization metadata: **Untouched**.
- Phase 12C synchronization reader: **Untouched**.
- Phase 10 temporal verification: **Strictly independent**.
- Phase 12D tracking and rider identity protection: **Preserved**.
- Model weights: **No training or fine-tuning performed**.
- Next phases: Phase 12F and Phase 12G have **NOT** been started.
