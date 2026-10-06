# M2.2 daily log

One short entry per working day, written on the day.

## 2026-10-06
- Plan for the checkpoint written; baseline fixed as the J1 end state (`d3eae82`).
- Moved loading/split, feature building and evaluation helpers out of notebook 01 into the `fraud/` package, with pytest tests (data, features incl. a leakage guard, metrics).
- Added `fraud/model.py` (hyper-parameters read from the J1 search results) and the experiment-log helper; slow test confirms the package reproduces the committed baseline.
- Notebook 03: pre-registered the decision rules, reproduced the baseline exactly (test PR-AUC 0.9972, 4 FN, 0 FP) and measured the seed noise floor, σ = 0.00026 validation PR-AUC.
- Next: look at the baseline's validation errors, then iteration 1.
- Looked at the baseline's 5 validation misses before choosing features: all are emptied-origin transactions where the amount is not the old balance. Defined an expected-cost metric (missed fraud money + review cost per alert) with tests and scored the baseline: it misses 2.31M on validation, 95% of its cost at 100 units per alert.
- Iteration 1 (ratio and flag features): validation PR-AUC +0.00012 against a 2σ bar of 0.00052, same 5 misses, so reverted by the rule. The model needed half the trees, because the new equality flag is a one-column version of a test the baseline already built. The misses are in a corner (916k rows, 20 frauds) that own-row features cannot separate.
