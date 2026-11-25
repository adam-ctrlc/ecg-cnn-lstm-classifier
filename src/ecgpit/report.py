"""Summarize all runs and draw the figures of the deployed model into results/.

    python -m ecgpit.report
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from .config import CLASSES, NAMES, ROOT


def main():
    out = ROOT / "results"; out.mkdir(exist_ok=True)
    rows = []
    for f in sorted((ROOT / "runs").glob("*/metrics.json")):
        m = json.loads(f.read_text()); t = m["test"]
        rows.append(dict(run=f.parent.name, model=m["model"], input=m["input"], train_records=m["train_records"], seed=m["seed"], params=m["params"], epochs=m["epochs_run"],
                         val_auc=m["best_val_macro_auc"], **{k: t["macro"][k] for k in t["macro"]}, micro_f1=t["micro_f1"],
                         exact_match=t["exact_match"], label_accuracy=t["label_accuracy"], **{f"auc_{c}": t["per_class"][c]["roc_auc"] for c in CLASSES}))
    df = pd.DataFrame(rows); df.to_csv(out / "all_runs.csv", index=False)
    num = [c for c in df.columns if c not in ("run", "model", "input", "train_records", "seed", "params")]
    g = df.groupby(["model", "input", "train_records"])[num].agg(["mean", "std"]); g.columns = ["_".join(c) for c in g.columns]
    g.insert(0, "runs", df.groupby(["model", "input", "train_records"]).size()); g.to_csv(out / "summary.csv")
    print(g[["runs", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "f1_mean", "f1_std", "auc_HYP_mean", "auc_HYP_std"]].round(4).to_string())

    info = json.loads((ROOT / "models" / "model.json").read_text()); run = ROOT / "runs" / info["run"]
    m = json.loads((run / "metrics.json").read_text()); z = np.load(run / "predictions.npz")
    y, p = z["y_test"], z["p_test"]
    pd.DataFrame(m["test"]["per_class"]).T.to_csv(out / "per_class.csv")
    h = pd.read_csv(run / "history.csv"); fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(h.epoch, h.train_loss, label="training"); ax[0].plot(h.epoch, h.val_loss, label="validation")
    ax[0].set(title="Binary cross-entropy", xlabel="Epoch", ylabel="Loss"); ax[0].legend()
    ax[1].plot(h.epoch, h.val_macro_auc, color="C1"); b = h.val_macro_auc.idxmax()
    ax[1].axvline(h.epoch[b], ls="--", color="gray"); ax[1].set(title="Validation macro ROC-AUC", xlabel="Epoch", ylabel="AUC")
    fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=200); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for i, c in enumerate(CLASSES):
        a, b_, _ = roc_curve(y[:, i], p[:, i]); ax[0].plot(a, b_, label=f"{c} {m['test']['per_class'][c]['roc_auc']:.3f}")
        q, r, _ = precision_recall_curve(y[:, i], p[:, i]); ax[1].plot(r, q, label=f"{c} {m['test']['per_class'][c]['pr_auc']:.3f}")
    ax[0].plot([0, 1], [0, 1], "k:", lw=0.8); ax[0].set(title="ROC, test fold", xlabel="False positive rate", ylabel="True positive rate")
    ax[1].set(title="Precision-recall, test fold", xlabel="Recall", ylabel="Precision")
    [a.legend(fontsize=8) for a in ax]; fig.tight_layout(); fig.savefig(out / "roc_pr.png", dpi=200); plt.close(fig)
    fig, ax = plt.subplots(1, 5, figsize=(13, 3))
    for a, c in zip(ax, CLASSES):
        k = m["test"]["per_class"][c]; cm = np.array([[k["tn"], k["fp"]], [k["fn"], k["tp"]]]); a.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                a.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        a.set(title=f"{c}", xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", xticklabels=["no", "yes"], yticklabels=["no", "yes"])
    ax[0].set_ylabel("Reference"); fig.tight_layout(); fig.savefig(out / "confusion_matrices.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
