"""
metrics.py
==========
Every metric named in Section 9, computed from raw logits/labels so both
train.py (per-epoch validation) and evaluate.py (final test report) share
one source of truth.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc,
)


def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=1)


def compute_core_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    """Accuracy, precision/recall/F1 (macro + weighted), per-class accuracy."""
    acc = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.diag(cm) / cm.sum(axis=1)
    per_class_acc = np.nan_to_num(per_class_acc)

    return {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "per_class_accuracy": per_class_acc.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def compute_topk_accuracy(y_true: np.ndarray, y_probs: np.ndarray, k_values=(1, 3, 5)) -> dict:
    """
    Manual top-k accuracy: for each sample, checks whether the true label
    is among the k highest-probability classes.

    NOTE: this is implemented by hand rather than via sklearn's
    `top_k_accuracy_score`, which raises a ValueError on binary
    classification (2-class y_true with a 2D y_score) even when `labels=`
    is explicitly supplied -- a sklearn quirk, not a real ambiguity in what
    "top-k accuracy" should mean here. The manual version below gives the
    same answer as sklearn for num_classes > 2 and additionally works for
    the num_classes == 2 case (exercised by our unit tests).
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    num_classes = y_probs.shape[1]
    out = {}
    for k in k_values:
        k_eff = min(k, num_classes)
        top_k_preds = np.argsort(-y_probs, axis=1)[:, :k_eff]
        correct = np.any(top_k_preds == y_true.reshape(-1, 1), axis=1)
        out[f"top_{k}_accuracy"] = float(np.mean(correct))
    return out


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: list) -> dict:
    return classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)


def expected_calibration_error(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 15) -> dict:
    """
    ECE: bins predictions by confidence, and within each bin compares mean
    confidence to actual accuracy. A well-calibrated model has confidence
    ~= accuracy in every bin (ECE ~ 0). Needed because Section 9 explicitly
    requires reporting "prediction confidence" -- confidence numbers are
    meaningless without checking whether they're calibrated.

    Returns per-bin data (for plotting a reliability diagram) plus the
    scalar ECE.
    """
    confidences = y_probs.max(axis=1)
    predictions = y_probs.argmax(axis=1)
    accuracies = (predictions == y_true).astype(np.float32)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_confidences, bin_accuracies, bin_counts = [], [], []
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        count = in_bin.sum()
        if count > 0:
            bin_acc = accuracies[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            ece += (count / n) * abs(bin_acc - bin_conf)
        else:
            bin_acc, bin_conf = 0.0, 0.0
        bin_confidences.append(float(bin_conf))
        bin_accuracies.append(float(bin_acc))
        bin_counts.append(int(count))

    return {
        "ece": float(ece),
        "bin_edges": bin_edges.tolist(),
        "bin_confidences": bin_confidences,
        "bin_accuracies": bin_accuracies,
        "bin_counts": bin_counts,
    }


def compute_roc_pr_curves(y_true: np.ndarray, y_probs: np.ndarray, num_classes: int) -> dict:
    """One-vs-rest ROC and PR curves, macro-averaged (Section 9, [NEW])."""
    y_true_onehot = np.eye(num_classes)[y_true]

    roc_data, pr_data = {}, {}
    aucs, pr_aucs = [], []

    for c in range(num_classes):
        if y_true_onehot[:, c].sum() == 0:
            continue  # class absent from this eval set; skip rather than error
        fpr, tpr, _ = roc_curve(y_true_onehot[:, c], y_probs[:, c])
        roc_auc_val = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, c], y_probs[:, c])
        pr_auc_val = auc(recall, precision)

        roc_data[c] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc_val}
        pr_data[c] = {"precision": precision.tolist(), "recall": recall.tolist(), "auc": pr_auc_val}
        aucs.append(roc_auc_val)
        pr_aucs.append(pr_auc_val)

    return {
        "per_class_roc": roc_data,
        "per_class_pr": pr_data,
        "macro_roc_auc": float(np.mean(aucs)) if aucs else float("nan"),
        "macro_pr_auc": float(np.mean(pr_aucs)) if pr_aucs else float("nan"),
    }


def most_confused_pairs(cm: np.ndarray, class_names: list, top_n: int = 10) -> list:
    """Returns the top_n (true_class, predicted_class, count) off-diagonal
    confusions, sorted descending -- directly feeds Section 10's error
    analysis and Section 10 [NEW]'s Grad-CAM sampling."""
    pairs = []
    n = cm.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append((class_names[i], class_names[j], int(cm[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_n]


if __name__ == "__main__":
    # Sanity check against known toy inputs (also exercised in tests/test_metrics.py)
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 0, 2, 2])
    metrics = compute_core_metrics(y_true, y_pred, num_classes=3)
    assert metrics["accuracy"] == 4 / 6
    print("metrics.py OK -- core metrics match hand-computed toy example.")
