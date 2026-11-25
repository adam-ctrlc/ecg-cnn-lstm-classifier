import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score, hamming_loss, precision_recall_fscore_support,
                             precision_recall_curve, roc_auc_score)

from .config import CLASSES


def macro_auc(y, p):
    return float(np.mean([roc_auc_score(y[:, i], p[:, i]) for i in range(y.shape[1])]))


def best_thresholds(y, p):
    """Per-class threshold with the highest F1. Called on the validation fold only."""
    out = []
    for i in range(y.shape[1]):
        pr, rc, th = precision_recall_curve(y[:, i], p[:, i])
        f1 = 2 * pr[:-1] * rc[:-1] / np.maximum(pr[:-1] + rc[:-1], 1e-9)
        out.append(float(np.clip(th[int(np.argmax(f1))], 0.05, 0.95)))
    return np.array(out)


def evaluate(y, p, thr, boot=0):
    h = (p >= thr).astype(int)
    pr, rc, f1, sup = precision_recall_fscore_support(y, h, average=None, zero_division=0)
    per = {}
    for i, c in enumerate(CLASSES):
        tn, fp, fn, tp = confusion_matrix(y[:, i], h[:, i], labels=[0, 1]).ravel()
        per[c] = dict(precision=float(pr[i]), recall=float(rc[i]), specificity=float(tn / max(tn + fp, 1)), f1=float(f1[i]),
                      roc_auc=float(roc_auc_score(y[:, i], p[:, i])), pr_auc=float(average_precision_score(y[:, i], p[:, i])),
                      support=int(sup[i]), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    keys = ("precision", "recall", "specificity", "f1", "roc_auc", "pr_auc")
    out = dict(n=len(y), per_class=per, macro={k: float(np.mean([per[c][k] for c in CLASSES])) for k in keys},
               micro_f1=float(f1_score(y, h, average="micro", zero_division=0)), exact_match=float(accuracy_score(y, h)),
               label_accuracy=float(1 - hamming_loss(y, h)))
    if boot:
        rng, vals = np.random.default_rng(0), []
        while len(vals) < boot:
            s = rng.integers(0, len(y), len(y))
            if (y[s].sum(0) > 0).all():
                vals.append(macro_auc(y[s], p[s]))
        out["macro_auc_ci95"] = [round(float(v), 4) for v in np.percentile(vals, [2.5, 97.5])]
    return out
