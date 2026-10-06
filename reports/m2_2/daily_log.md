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
- Iteration 2 (cost-based threshold, same scores): threshold 0.506 to 0.041, 3 missed frauds instead of 5, 589 extra false alarms; validation cost at 100 units per alert down 27% in-sample and 13% cross-fitted. Kept, with the caveat that the threshold rests on 5 misses; the test set has not been touched yet.
- Iteration 3 (second-stage search, 12 configs, 3-fold CV on the full training set, 25 minutes of compute): the CV winner beat the baseline configuration in CV (0.99812 vs 0.99681) but not on validation (PR-AUC -0.00006, more false alarms, higher cost), so reverted. Tuning is saturated, as J1 found.
- Problem fix: the raw-feature ablation model needed 1,886 of 2,000 trees. Refit with learning rate 0.1: 733 trees, same accuracy (test PR-AUC 0.9295 vs 0.9289), so the J1 ablation gap is not an under-training artefact. Notebook 04.
