"""
visualization.py
================
Every plot named in Section 11, each with a title, axis labels, and legend
(the "professional formatting" requirement). Every function saves a PNG to
figures/ rather than displaying interactively, so this is scriptable in
headless training environments too.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _finalize(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curves(history: list, save_path: str):
    """Training/validation loss AND accuracy/F1 curves in one figure."""
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Train Loss")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="Val Loss")
    axes[0].set_title("Loss vs. Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, [h["train_f1_macro"] for h in history], label="Train Macro-F1")
    axes[1].plot(epochs, [h["val_f1_macro"] for h in history], label="Val Macro-F1")
    axes[1].set_title("Macro-F1 vs. Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1 Score")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    _finalize(fig, save_path)


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str, normalize: bool = True):
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_display = np.nan_to_num(cm_display)
        fmt_label = "Normalized Confusion Matrix (row = true class)"
    else:
        cm_display = cm
        fmt_label = "Confusion Matrix (raw counts)"

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm_display, cmap="Blues")
    ax.set_title(fmt_label)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _finalize(fig, save_path)


def plot_class_distribution(class_counts: dict, save_path: str):
    names = list(class_counts.keys())
    counts = list(class_counts.values())
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(names, counts)
    ax.set_title("Class Distribution in Dataset")
    ax.set_xlabel("Class (ASL Letter / Token)")
    ax.set_ylabel("Number of Images")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.grid(axis="y", alpha=0.3)
    _finalize(fig, save_path)


def plot_hyperparameter_comparison(results: dict, param_name: str, metric_name: str, save_path: str):
    """
    results: {param_value: [metric_seed1, metric_seed2, metric_seed3], ...}
    Plots mean +/- std as an error-bar chart -- used for Exp 1-5 comparisons
    (LR, batch size, dropout, optimizer, depth), consistent with the
    multi-seed statistical-validity requirement in Section 8 [NEW].
    """
    labels = list(results.keys())
    means = [np.mean(v) for v in results.values()]
    stds = [np.std(v) for v in results.values()]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, means, yerr=stds, capsize=6)
    ax.set_title(f"{metric_name} by {param_name} (mean \u00b1 std over seeds)")
    ax.set_xlabel(param_name)
    ax.set_ylabel(metric_name)
    ax.grid(axis="y", alpha=0.3)
    _finalize(fig, save_path)


def plot_reliability_diagram(calibration: dict, save_path: str):
    """Confidence vs. accuracy calibration plot (Section 9/11, [NEW])."""
    bin_edges = calibration["bin_edges"]
    centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax.bar(
        centers,
        calibration["bin_accuracies"],
        width=1.0 / len(centers),
        alpha=0.7,
        edgecolor="black",
        label="Model Accuracy",
    )
    ax.set_title(f"Reliability Diagram (ECE = {calibration['ece']:.4f})")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    _finalize(fig, save_path)


def plot_roc_curves(roc_pr: dict, class_names: list, save_path: str, max_classes_shown: int = 10):
    """Plots ROC curves for up to `max_classes_shown` classes (all 29 on one
    plot is unreadable) plus reports the macro-average AUC in the title."""
    fig, ax = plt.subplots(figsize=(7, 7))
    per_class = roc_pr["per_class_roc"]
    shown = list(per_class.keys())[:max_classes_shown]

    for c in shown:
        data = per_class[c]
        ax.plot(data["fpr"], data["tpr"], label=f"{class_names[int(c)]} (AUC={data['auc']:.2f})", alpha=0.7)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title(f"ROC Curves, One-vs-Rest (macro AUC = {roc_pr['macro_roc_auc']:.4f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=6, loc="lower right")
    _finalize(fig, save_path)


def plot_first_layer_filters(model, save_path: str):
    """Visualizes the learned filters of the FIRST conv layer -- shows
    whether the network learned sensible low-level edge/color detectors."""
    first_conv = model.features[0].conv
    weights = first_conv.weight.detach().cpu().numpy()  # (out_ch, in_ch, k, k)
    n_filters = min(weights.shape[0], 32)

    fig, axes = plt.subplots(4, n_filters // 4, figsize=(n_filters // 4 * 1.2, 5))
    for i, ax in enumerate(axes.flat):
        if i >= n_filters:
            ax.axis("off")
            continue
        w = weights[i].transpose(1, 2, 0)
        w = (w - w.min()) / (w.max() - w.min() + 1e-8)
        ax.imshow(w)
        ax.axis("off")
    fig.suptitle("Learned First-Layer Convolutional Filters")
    _finalize(fig, save_path)


def plot_training_time_comparison(time_per_config: dict, save_path: str):
    labels = list(time_per_config.keys())
    times = list(time_per_config.values())
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, times)
    ax.set_title("Training Time by Configuration")
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Wall-Clock Time (minutes)")
    ax.grid(axis="y", alpha=0.3)
    _finalize(fig, save_path)
