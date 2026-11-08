# Phase 12E — Review Correction 2: Prevention of Below-Threshold Candidate Blocking in Global Assignment

## 1. Executive Summary

This report documents the resolution of the **Phase 12E Review Correction 2** defect in `src/association/global_assignment.py`.

During researcher review of commit `004a8b7e2e3f5383501a3574c860c0429f582ae4`, an assignment-integrity vulnerability was identified: although Review Correction 1 correctly excluded failed-hard-gate candidates from the Hungarian cost matrix, candidates with `composite_score < min_association_score` (sub-threshold candidates) were still assigned valid status (`valid_mask = True`). Consequently, in the augmented cost matrix, a sub-threshold candidate (e.g., \(S=0.61 < 0.62\)) had a lower cost (\(1.0 - 0.61 = 0.39\)) than leaving tracklets unmatched (cost \(2.0\)). Hungarian optimization paired this sub-threshold candidate, consuming a tracklet and displacing viable alternative pairings that could have achieved a valid `LINKED` assignment (such as a 2-pair matching). The sub-threshold pair was subsequently discarded by post-filtering (`if best_score < min_association_score: continue`), leaving both tracklets permanently unmatched.

This defect has been resolved by:
1. Extending candidate optimization eligibility so that only candidates satisfying all hard gates, sufficient active cues, **and** `round(composite_score, 6) >= round(min_association_score, 6)` (inclusive) enter the augmented Hungarian optimization matrix.
2. Preventing sub-threshold candidates from occupying real tracklets or blocking feasible assignments of linkable candidates.
3. Isolating ambiguity calculations: sub-threshold candidates cannot serve as competitors in row/column uniqueness margins or as cross-terms in 2×2 swap margins, preventing false `AMBIGUOUS_ABSTAIN` outcomes.
4. Preserving complete diagnostic provenance: sub-threshold candidate scores, cue breakdowns, and rejection reasons (`BEST_SCORE_X_BELOW_MIN_0.62`) remain available in audit outputs and unassigned tracklet results.
5. Implementing 4 new regression tests (Tests F, G, H, I) in `tests/test_phase12e_association.py`.

The full repository test suite passed with 85 passed tests (0 failures, 5 non-fatal library deprecation warnings).

---

## 2. Root Cause Analysis

In the Review Correction 1 implementation:
```python
# PREVIOUS CODE:
if (
    cs is not None
    and cs.passed_candidate_gate
    and cs.active_cues_count >= self.config.min_active_cues_required
):
    valid_mask[i, j] = True
    valid_score_mat[i, j] = float(cs.composite_score)
```

In the augmented Hungarian cost matrix:
\[
C[i, j] = 1.0 - S_{i, j} \quad \text{for } \text{valid\_mask}[i, j] = \text{True}
\]
Because remaining unmatched cost is \(1.0 + 1.0 = 2.0\), any candidate with \(S_{i, j} > 0\) incurred cost \(1.0 - S_{i, j} < 1.0\), yielding positive savings \(1.0 + S_{i, j} \ge 1.0 > 0\).

Consider the researcher's benchmark matrix (\(\tau = 0.62\)):

| | B1 | B2 |
|---|---:|---:|
| **F1** | 0.61 | 0.62 |
| **F2** | 0.62 | 0.99 |

- Hungarian evaluated:
  - Option 1: \((F1, B1)\) [\(S=0.61\)] and \((F2, B2)\) [\(S=0.99\)]. Sum = \(1.60\). Cost = \(0.39 + 0.01 = 0.40\).
  - Option 2: \((F1, B2)\) [\(S=0.62\)] and \((F2, B1)\) [\(S=0.62\)]. Sum = \(1.24\). Cost = \(0.38 + 0.38 = 0.76\).
- Hungarian chose Option 1 because \(0.40 < 0.76\).
- Post-processing then discarded \((F1, B1)\) because \(0.61 < 0.62\).
- Result: \(F1\) and \(B1\) were left `UNMATCHED`, and the viable 2-pair linkable assignment \(\{ (F1, B2), (F2, B1) \}\) was permanently lost.

---

## 3. Mathematical Assignment Policy & Threshold Semantics

### 3.1 Optimization Eligibility Threshold
A candidate pair \((i, j)\) is defined as eligible for Hungarian matching and margin competition if and only if:
\[
\text{Eligible}(i, j) \iff \begin{cases}
\text{cs} \ne \text{None} \\
\text{cs.passed\_candidate\_gate} = \text{True} \\
\text{cs.active\_cues\_count} \ge \text{min\_active\_cues\_required} \\
\text{round}(\text{cs.composite\_score}, 6) \ge \text{round}(\tau, 6)
\end{cases}
\]
where \(\tau = \text{min\_association\_score}\) (default \(0.62\)).

### 3.2 Inclusive Threshold Boundary Semantics
The comparison is strictly **INCLUSIVE**:
- A candidate with score \(S < \tau\) (e.g. \(0.6199 < 0.6200\)) is **ineligible** (`valid_mask = False`). It cannot enter optimization, cannot block other candidates, cannot create ambiguity, and cannot link.
- A candidate with score \(S = \tau\) (e.g. \(0.6200 = 0.6200\)) is **eligible** (`valid_mask = True`). If uniqueness margins allow, it receives `LINKED`.
- A candidate with score \(S > \tau\) (e.g. \(0.6201 > 0.6200\)) is **eligible** (`valid_mask = True`).

Rounding to 6 decimal places (`round(..., 6)`) guarantees IEEE 754 floating-point consistency across Python environments, eliminating binary representation drift.

### 3.3 Prevention of Blocking
In the benchmark matrix:
- \((F1, B1)\) has \(0.61 < 0.62 \implies \text{valid\_mask}[0, 0] = \text{False}\). Cost in augmented matrix = \(10^6\).
- \((F1, B2)\), \((F2, B1)\), \((F2, B2)\) are all \(\ge 0.62 \implies \text{valid\_mask} = \text{True}\).
- Hungarian optimization evaluates feasible matchings:
  - 2-pair matching: \((F1, B2)\) and \((F2, B1)\). Cost = \(0.38 + 0.38 = 0.76\).
  - 1-pair matching: \((F2, B2)\), with \(F1\) and \(B1\) unmatched. Cost = \(0.01 + 1.0 + 1.0 = 2.01\).
- Since \(0.76 < 2.01\), Hungarian chooses \(\{ (F1, B2), (F2, B1) \}\).
- Thus, the sub-threshold candidate \(F1 \to B1\) is prevented from blocking \(F1 \to B2\), and \(F2 \to B2\) does not prevent a higher-cardinality feasible assignment.

### 3.4 Margin Isolation
- Sub-threshold candidates are excluded from `row_competitors` and `col_competitors`. A candidate with \(S = 0.60 < 0.62\) cannot lower the uniqueness margin of a valid candidate with \(S = 0.65\).
- Valid above-threshold alternatives (\(S \ge 0.62\)) remain active competitors. If \(S_1 = 0.70\) and alternative \(S_2 = 0.65\), the margin \(0.70 - 0.65 = 0.05 < 0.10\) correctly triggers `AMBIGUOUS_ABSTAIN`.

### 3.5 Provenance Retention
When an unassigned tracklet has no eligible candidate above threshold:
- `outcome = AssociationOutcome.UNMATCHED`
- `association_score = best_sc` (e.g., \(0.610\))
- `decision_reasons = [f"BEST_SCORE_{best_sc:.3f}_BELOW_MIN_{self.config.min_association_score}"]`
- `pair_score_detail` preserves the full multi-cue diagnostic record from `score_lookup`.

---

## 4. Regression Test Results

Four new regression tests (Tests F–I) were added to `tests/test_phase12e_association.py`:

| Test Identifier | Description | Outcome |
| :--- | :--- | :--- |
| **Test F** (`test_review_correction_test_f_...`) | Hand-constructed matrix \([[0.61, 0.62], [0.62, 0.99]]\). Sub-threshold \(F1 \to B1\) (\(0.61\)) cannot block \(F1 \to B2\) (\(0.62\)). Hungarian pairs \((F1, B2)\) and \((F2, B1)\). No candidate below threshold links or receives a global ID. | **PASSED** |
| **Test G** (`test_review_correction_test_g_...`) | All candidate pairs below threshold (\(0.50, 0.55, 0.60, 0.61 < 0.62\)). No links made; all tracklets `UNMATCHED`; decision reasons preserve `BELOW_MIN` provenance. | **PASSED** |
| **Test H** (`test_review_correction_test_h_...`) | Threshold boundary semantics: \(0.6199\) is `UNMATCHED`, \(0.6200\) is `LINKED` (inclusive), and \(0.6201\) is `LINKED`. | **PASSED** |
| **Test I** (`test_review_correction_test_i_...`) | Below-threshold alternative (\(0.60 < 0.62\)) cannot create false ambiguity on valid \(0.65\) (links cleanly); valid above-threshold alternative (\(0.65 \ge 0.62\)) triggers `AMBIGUOUS_ABSTAIN` on \(0.70\) (\(\Delta = 0.05 < 0.10\)). | **PASSED** |

### Test Suite Execution
- **Phase 12E Test Suite (`pytest tests/test_phase12e_association.py -v`):**
  - **35 / 35 passed** in 3.54s (0 failures).
- **Full Repository Test Suite (`/opt/homebrew/bin/python3.11 -m pytest tests/ -v`):**
  - **85 / 85 passed**, 5 library deprecation warnings, 0 failures.

---

## 5. Scope Preservation Confirmation
- Original video files: **Untouched**.
- Phase 11.6 synchronization metadata: **Untouched**.
- Phase 12C synchronization reader: **Untouched**.
- Phase 10 temporal verification: **Strictly independent**.
- Phase 12D rider identity protection: **Preserved**.
- Model weights: **No training or fine-tuning performed**.
- Next phases: Phase 12F and Phase 12G have **NOT** been started.
