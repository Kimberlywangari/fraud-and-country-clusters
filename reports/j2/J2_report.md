# J2: evaluation and hyper-parameter tuning of the fraud model

The question: is the fraud model actually good, measured the right way, and does tuning it with a real search help? The work is in [notebook 06](../../notebooks/06_evaluation_and_tuning.ipynb), and the leakage write-up is in [leakage_note.md](leakage_note.md).

## What was tuned, and how

| | XGBoost | Logistic regression |
|---|---|---|
| Method | `RandomizedSearchCV`, 40 random configurations | `GridSearchCV`, all 10 combinations |
| Searched | trees, depth, learning rate, subsample, column sample, min child weight, gamma, L2, class weight | regularisation strength C, class weight |
| Scored by | 3-fold stratified CV, PR-AUC, on a stratified 30% sample of the training rows | same |
| Untuned = | library defaults (depth 6, learning rate 0.3, 100 trees, no class weight) | C = 1, no class weight |
| Best found | depth 5, learning rate 0.05, 100 trees, full class weight (336) | C = 0.01, no class weight |

The searches see training rows only (asserted in the notebook). Neither model uses early stopping in this comparison, so the only difference between tuned and untuned is the hyper-parameters.

## Tuned against untuned

Test set (415,562 transactions, 1,232 frauds), thresholds chosen on the validation set. CV is stratified 5-fold on the training set, mean with standard deviation across folds.

| Model | CV PR-AUC | Test PR-AUC | Test ROC-AUC | Precision | Recall | F1 | Missed frauds | False alarms |
|---|---|---|---|---|---|---|---|---|
| XGBoost, untuned | 0.9964 ± 0.0028 | 0.9975 | 0.9983 | 1.000 | 0.994 | 0.997 | 7 | 0 |
| XGBoost, tuned | 0.9980 ± 0.0013 | 0.9975 | 0.9987 | 1.000 | 0.997 | 0.998 | 4 | 0 |
| Logistic regression, untuned | 0.9925 ± 0.0026 | 0.9921 | 0.9954 | 0.990 | 0.982 | 0.986 | 22 | 12 |
| Logistic regression, tuned | 0.9927 ± 0.0025 | 0.9922 | 0.9954 | 0.985 | 0.984 | 0.985 | 20 | 18 |

Is the tuned-minus-untuned PR-AUC difference real? Paired bootstrap on the test set, 1,000 resamples:

| Model | Difference | 95% interval | Reading |
|---|---|---|---|
| XGBoost | −0.00003 | −0.00011 to +0.00001 | not distinguishable from noise |
| Logistic regression | +0.00013 | −0.00000 to +0.00038 | not distinguishable from noise, only just |

## What this shows

1. **Tuning did not improve how well either model ranks frauds.** The test PR-AUC is the same before and after for XGBoost, and the bootstrap interval straddles zero for both models. In cross-validation tuned XGBoost looks better (0.9980 against 0.9964), but the untuned model's fold-to-fold spread (0.0028) is larger than that gap, and the tuned CV score is somewhat optimistic anyway: the CV runs on training rows that include the sample the search used to choose the settings. The test set, which the search never saw, is the fair comparison. This agrees with what J1 and the M2.2 work found: this model is near its ceiling on this data.
2. **Tuning did move XGBoost's operating point.** At the validation-chosen threshold the untuned model misses 7 frauds and the tuned one 4, with no false alarms in either. The untuned model's best-F1 threshold is extreme (0.9995) because without class weighting its scores are squeezed towards zero. The counts are small and I did not test them, so I read this as suggestive.
3. **A correction to J1.** The grid search turned class weighting off for logistic regression; "balanced" was worse at every C. J1's logistic regression used "balanced", so it was a weaker baseline than necessary. Tuned, it scores 0.992 on the test set instead of 0.970, and XGBoost's lead falls from about 0.028 to about 0.005. XGBoost is still ahead, by less than J1 said.

## Why these metrics

The data is 0.3% fraud, so the choice of metric decides whether the evaluation means anything.

| Metric | Role | Why |
|---|---|---|
| Accuracy | not used | A model that never flags anything is 99.70% accurate on the test set and catches no fraud. |
| ROC-AUC | reported, not the headline | Logistic regression on the raw features has ROC-AUC 0.992 but PR-AUC 0.549 (`reports/ablation.csv`): it looks excellent while about half its alerts would be wrong. The very large number of legitimate rows flatters it. |
| PR-AUC (average precision) | headline | It asks how precise the alerts are at each level of recall, which is what an analyst feels. A guessing model scores the fraud rate (0.003), so the scale is honest. |
| Precision, recall, F1, missed frauds, false alarms | operating point | They need a threshold, chosen on validation and applied unchanged to test. The counts say how many frauds get through and how many alerts are wasted. |
| Stratified CV | how searching and checking are run | Plain folds could hold very few frauds. Stratified folds keep the 0.3% rate in every fold. |

## Leakage

Detail in [leakage_note.md](leakage_note.md). In short: features do not use the label; no single column gives it away (the best reaches 0.29 PR-AUC alone); the 11 test frauds that have an identical training row do not change the score; shuffling the training labels leaves the score at no more than 5.4% of its real value; and the search, the thresholds and the scaler stayed on the rows they were allowed to see. The model relies on balances recorded after a transaction settles, and without them it scores 0.76 instead of 0.9975, so the headline describes after-the-fact detection.

## Tests

`python -m pytest -q` runs 65 tests in a few seconds on small synthetic frames. The J2 additions:
- `tests/test_search.py` (7): the scoring metric is average precision; folds and subsamples keep the fraud count; a grid tries every combination; the reported score equals a hand-run stratified `cross_val_score`; a random search draws the requested number of configurations from the stated space, repeats for a fixed seed and differs across seeds.
- `tests/test_leakage.py` (11): label-independence of the features, including a planted label-derived feature that is caught; duplicate detection with planted twins and with near-twins that differ in one column; the allowed-rows guard; shuffled labels fall for an honest pipeline and stay high for one that peeks at the test labels; the single-feature screen ranks a planted leak first whatever its sign; the scaler holds training-set statistics only.
- `tests/test_model.py` (2 new): `signed_log1p` keeps the sign and works in float64 even for int8 columns.

## A defect found along the way

`signed_log1p`, moved from the J1 notebook, silently computed int8 columns (`isTransfer`, `origEmptied`, `hour`) in float16. The effect on the model was tiny, but it made my own scaler check return NaN and fail, which is how it was found. It now casts to float64 first, with a test. The J1 notebook is left as committed.

## Limits

- The tuned CV scores overlap with the search sample (see above), so only the test comparison is unbiased.
- Search used a 30% training sample, and PaySim is synthetic, so the scores say more about the simulator than about real fraud.
- One dataset, one split, one seed for the final fits. The bootstrap and the CV spread cover sampling noise, not that.
- The model is a post-settlement detector (see the leakage note).
- No time-based check was repeated for the tuned model.

## Reproducing

```
python -m pytest -q
python -m nbconvert --to notebook --execute --inplace notebooks/06_evaluation_and_tuning.ipynb
```

The notebook takes about 15 minutes. Seeds are fixed.
