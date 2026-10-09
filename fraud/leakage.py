"""Data-leakage checks (J2). Each function answers one specific question, and says what passing does NOT prove.

1. features_ignore_label     - are the features independent of the label column? (rules out label-derived features)
2. rows_shared               - do identical feature rows sit in both train and test? (rules out memorised duplicates)
3. check_rows_within         - did a step use only the rows it was allowed to? (rules out search/threshold on test rows)
4. permuted_label_scores     - with the training labels shuffled, does the test score fall to chance? (rules out any
                               route by which test information reaches the score without passing through the labels
                               the model was trained on)
5. univariate_pr_auc         - is any single feature suspiciously good on its own? (a screen, not a proof)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fraud.data import SEED


class LeakageError(AssertionError):
    """Raised when a check finds rows or information where it should not be."""


def features_ignore_label(build_features, frame, label_col, seed=SEED):
    """True if the features come out identical after the label column is shuffled.

    `build_features(frame)` must return the feature table (first element if it returns a tuple). If the features
    used the label in any way, shuffling it would change them.
    """
    def features(f):
        out = build_features(f)
        return out[0] if isinstance(out, tuple) else out

    shuffled = frame.assign(**{label_col: np.random.default_rng(seed).permutation(frame[label_col].to_numpy())})
    return features(frame).equals(features(shuffled))


def row_hashes(X):
    """One 64-bit hash per feature row; equal rows get equal hashes."""
    return pd.util.hash_pandas_object(X, index=False).to_numpy()


def rows_shared(X_reference, X_query):
    """Boolean mask over X_query: True where an identical feature row also appears in X_reference.

    Identical inputs get identical predictions, so a query row with a twin in the reference set (typically the
    training set) can be answered by memory instead of by what the model learned.
    """
    return np.isin(row_hashes(X_query), row_hashes(X_reference))


def check_rows_within(rows, allowed, name="rows"):
    """Raise LeakageError unless every position in `rows` is in `allowed` (e.g. search rows must be training rows)."""
    outside = np.setdiff1d(np.asarray(rows), np.asarray(allowed))
    if len(outside):
        raise LeakageError(f"{name}: {len(outside)} rows are outside the allowed set (first few: {outside[:5].tolist()})")


def permuted_label_scores(fit_predict, X_train, y_train, X_test, y_test, seeds=(SEED,)):
    """Average precision on the TRUE test labels after fitting on SHUFFLED training labels, one value per seed.

    `fit_predict(X_train, y_train, X_test)` runs the whole pipeline (fit, any early stopping or threshold choice that
    the real run uses) and returns test scores. With shuffled labels there is nothing real to learn, so the result
    should sit at the fraud rate. A result well above it means information about the test labels is reaching the
    score by some route that does not go through the training labels.
    """
    scores = []
    for seed in seeds:
        shuffled = np.random.default_rng(seed).permutation(y_train)
        scores.append(average_precision_score(y_test, fit_predict(X_train, shuffled, X_test)))
    return scores


def univariate_pr_auc(X, y):
    """PR-AUC of each feature used alone as a score (the better of +feature and -feature), best first.

    A value near 1 means one column almost gives the label away. That needs an explanation (is the column
    computed from the row itself, and available when the prediction is made?), but it is not proof of a leak.
    """
    out = {col: max(average_precision_score(y, X[col]), average_precision_score(y, -X[col])) for col in X.columns}
    return pd.Series(out).sort_values(ascending=False)
