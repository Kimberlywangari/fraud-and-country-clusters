"""Evaluation helpers, moved from notebooks/01_supervised_fraud.ipynb (cells 25 and 49) without changing behaviour.

Accuracy is useless at 0.3 % fraud, so PR-AUC (average precision) is the headline metric. Precision, recall and
F1 need a threshold, which is chosen on the validation set and then applied unchanged to the test set.
"""
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score


def best_f1_threshold(y_true, score):
    """Score threshold with the highest F1 on the given labelled set."""
    p, r, thr = precision_recall_curve(y_true, score)
    # precision_recall_curve returns one more (p, r) point than thresholds, hence the [:-1].
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(thr[np.argmax(f1)])


def evaluate(name, y_true, score, threshold, features="", fit_seconds=np.nan):
    """One row of metrics: threshold-free (PR-AUC, ROC-AUC) plus confusion-matrix metrics at `threshold`."""
    pred = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"model": name, "features": features,
            "PR-AUC": average_precision_score(y_true, score), "ROC-AUC": roc_auc_score(y_true, score),
            "precision": precision, "recall": recall, "F1": f1, "accuracy": (tp + tn) / len(y_true),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn), "fit_seconds": fit_seconds}


class WeightedAP:
    """Average precision under per-row integer weights; ties in the score are grouped (as in scikit-learn).

    Used by the paired bootstrap: each resample is a vector of Poisson(1) weights, one per test row, and the same
    weights are applied to every model so the comparison is paired. Sorting is done once in __init__.
    """

    def __init__(self, y_true, score):
        self.order = np.argsort(-score, kind="stable")
        s = score[self.order]
        self.y = y_true[self.order].astype(float)
        # Index of the last row of each run of equal scores: precision is only read after a whole tie group.
        self.ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]

    def __call__(self, w):
        ws = w[self.order]
        tp = np.cumsum(ws * self.y)[self.ends]
        seen = np.cumsum(ws)[self.ends]
        precision = np.divide(tp, seen, out=np.zeros_like(tp), where=seen > 0)
        gain = np.diff(np.r_[0.0, tp])
        return float(np.sum(gain / tp[-1] * precision))
