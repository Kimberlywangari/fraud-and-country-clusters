# M2.2 daily log

One short entry per working day, written on the day.

## 2026-10-06
- Plan for the checkpoint written; baseline fixed as the J1 end state (`d3eae82`).
- Moved loading/split, feature building and evaluation helpers out of notebook 01 into the `fraud/` package, with pytest tests (data, features incl. a leakage guard, metrics).
- Added `fraud/model.py` (hyper-parameters read from the J1 search results) and the experiment-log helper; slow test confirms the package reproduces the committed baseline.
- Notebook 03: pre-registered the decision rules, reproduced the baseline exactly (test PR-AUC 0.9972, 4 FN, 0 FP) and measured the seed noise floor, σ = 0.00026 validation PR-AUC.
- Next: look at the baseline's validation errors, then iteration 1.
