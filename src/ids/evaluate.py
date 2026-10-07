from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def predict_with_proba(model, Z, n_classes: int):
    raw = model.predict_proba(Z)
    cls = np.asarray(getattr(model, "classes_", np.arange(raw.shape[1])))
    proba = np.zeros((len(Z), n_classes), dtype=np.float32)
    proba[:, cls] = raw
    pred = cls[raw.argmax(axis=1)].astype(np.int64)
    return pred, proba


def _auc(y, proba, n_classes):
    try:
        if n_classes == 2:
            return float(roc_auc_score(y, proba[:, 1]))
        Y = np.asarray(label_binarize(y, classes=np.arange(n_classes)))
        s = Y.sum(0)
        present = np.flatnonzero((s > 0) & (s < len(y)))
        if len(present) == 0:
            return float("nan")
        scores = np.asarray([roc_auc_score(Y[:, k], proba[:, k]) for k in present])
        return float(np.mean(scores))
    except ValueError:
        return float("nan")


def compute_metrics(y_true, y_pred, proba, n_classes: int) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    fw = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[2]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "f1_weighted": float(fw),
        "auc": _auc(y_true, proba, n_classes),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def per_class_metrics(y_true, y_pred, class_names) -> "pd.DataFrame":
    """Return one-vs-rest metrics; recall is per-class accuracy/sensitivity."""
    labels = np.arange(len(class_names))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows = []
    for i, name in enumerate(class_names):
        true_positive = cm[i, i]
        false_positive = cm[:, i].sum() - true_positive
        false_negative = cm[i, :].sum() - true_positive
        true_negative = cm.sum() - true_positive - false_positive - false_negative
        specificity = (
            true_negative / (true_negative + false_positive)
            if true_negative + false_positive else 0.0
        )
        rows.append({
            "class_id": i,
            "class": name,
            "precision": float(precision[i]),
            "recall_class_accuracy": float(recall[i]),
            "f1": float(f1[i]),
            "specificity": float(specificity),
            "support": int(support[i]),
        })
    return pd.DataFrame(rows)


def plot_confusion_matrix(cm, names, path, title=""):
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(4, 0.7 * n + 2), max(4, 0.7 * n + 1.5)))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=90, fontsize=8); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label"); ax.set_title(title, fontsize=10)
    thr = cm.max() / 2 if cm.max() else 0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=6 if n > 5 else 10,
                    color="white" if cm[i, j] > thr else "black")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_roc_curves(entries: dict, n_classes: int, path, title=""):
    """entries: model -> (y_true, proba). Multiclass uses the micro-average curve."""
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (y, proba) in entries.items():
        try:
            if n_classes == 2:
                fpr, tpr, _ = roc_curve(y, proba[:, 1]); auc = roc_auc_score(y, proba[:, 1])
            else:
                Y = np.asarray(label_binarize(y, classes=np.arange(n_classes)))
                fpr, tpr, _ = roc_curve(Y.ravel(), proba.ravel()); auc = roc_auc_score(Y, proba, average="micro")
        except ValueError:
            continue
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc * 100:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title(title, fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
