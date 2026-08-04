"""
evaluate.py
===========
Final evaluation of a trained checkpoint on the held-out test set.

Usage:
    python evaluate.py --config configs/baseline.yaml --checkpoint checkpoints/efficientnet_v2_s_baseline/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import ExperimentConfig
from dataset import get_dataloaders
from model import get_model
from metrics import (
    compute_core_metrics, compute_topk_accuracy, get_classification_report,
    expected_calibration_error, compute_roc_pr_curves, most_confused_pairs,
)
from utils import (
    get_device, load_checkpoint, measure_inference_latency, get_git_commit_hash,
    count_parameters, model_size_mb, count_flops,
)
from visualization import plot_confusion_matrix, plot_reliability_diagram, plot_roc_curves
from logger import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def collect_predictions(model, loader, device):
    all_logits, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def run_evaluation(cfg: ExperimentConfig, checkpoint_path: str) -> dict:
    device = get_device(cfg.train.device)
    _, _, test_loader, _, class_names = get_dataloaders(cfg)
    cfg.data.class_names = class_names
    cfg.data.num_classes = len(class_names)

    model = get_model(
        cfg.model.architecture, num_classes=cfg.data.num_classes,
        pretrained=False,  # loading trained weights next -- no need to also download ImageNet weights first
        dropout=cfg.model.dropout,
    ).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()

    logits, labels = collect_predictions(model, test_loader, device)
    probs = F.softmax(logits, dim=1).numpy()
    preds = logits.argmax(dim=1).numpy()
    labels_np = labels.numpy()

    core = compute_core_metrics(labels_np, preds, cfg.data.num_classes)
    topk = compute_topk_accuracy(labels_np, probs)
    report = get_classification_report(labels_np, preds, class_names)
    calibration = expected_calibration_error(labels_np, probs)
    roc_pr = compute_roc_pr_curves(labels_np, probs, cfg.data.num_classes)
    confused = most_confused_pairs(np.array(core["confusion_matrix"]), class_names)

    input_shape = (3, cfg.data.image_size, cfg.data.image_size)
    latency = measure_inference_latency(model, input_shape, device)
    flops = count_flops(model, input_shape)
    efficiency = {"num_params": count_parameters(model), "model_size_mb": model_size_mb(model), **flops, **latency}

    figures_dir = Path("figures")
    figures_dir.mkdir(exist_ok=True)
    plot_confusion_matrix(np.array(core["confusion_matrix"]), class_names, figures_dir / f"{cfg.name}_confusion_matrix.png")
    plot_reliability_diagram(calibration, figures_dir / f"{cfg.name}_reliability_diagram.png")
    plot_roc_curves(roc_pr, class_names, figures_dir / f"{cfg.name}_roc_curves.png")

    result = {
        "experiment_name": cfg.name,
        "checkpoint": checkpoint_path,
        "git_commit": get_git_commit_hash(),
        "core_metrics": {k: v for k, v in core.items() if k != "confusion_matrix"},
        "confusion_matrix": core["confusion_matrix"],
        "top_k_accuracy": topk,
        "classification_report": report,
        "calibration": calibration,
        "roc_pr_macro": {"macro_roc_auc": roc_pr["macro_roc_auc"], "macro_pr_auc": roc_pr["macro_pr_auc"]},
        "most_confused_pairs": confused,
        "efficiency": efficiency,
    }

    out_path = Path("outputs") / f"{cfg.name}_test_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Test accuracy={core['accuracy']:.4f} | macro-F1={core['f1_macro']:.4f} | ECE={calibration['ece']:.4f}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = ExperimentConfig.load(args.config)
    run_evaluation(cfg, args.checkpoint)
