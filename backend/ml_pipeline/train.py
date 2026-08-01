"""
train.py
========
Training loop for ASLEfficientNetV2. Implements the staged fine-tuning
recipe described in model.py: the backbone is frozen for the first
`freeze_backbone_epochs`, training only the new classifier head, then
unfreezes for full fine-tuning with a lower backbone LR than the head LR.

Usage:
    python train.py --config configs/baseline.yaml
    python train.py --config configs/baseline.yaml --resume checkpoints/efficientnet_v2_s_baseline/last.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import ExperimentConfig
from dataset import get_dataloaders
from model import get_model
from metrics import compute_core_metrics
from utils import (
    set_seed, get_device, get_git_commit_hash, get_hardware_info, count_parameters, model_size_mb,
    save_checkpoint, load_checkpoint, EarlyStopping,
)
from logger import get_logger

logger = get_logger(__name__)


def build_optimizer(model, optim_cfg):
    """Two parameter groups: backbone (lower LR) and head (full LR) --
    the standard transfer-learning recipe so the pretrained backbone
    doesn't get scrambled by the same learning rate as the fresh head."""
    param_groups = [
        {"params": model.head_parameters(), "lr": optim_cfg.lr},
        {"params": model.backbone_parameters(), "lr": optim_cfg.lr * optim_cfg.backbone_lr_multiplier},
    ]
    if optim_cfg.optimizer == "sgd":
        return torch.optim.SGD(param_groups, momentum=0.9, weight_decay=optim_cfg.weight_decay)
    elif optim_cfg.optimizer == "adam":
        return torch.optim.Adam(param_groups, weight_decay=optim_cfg.weight_decay)
    elif optim_cfg.optimizer == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=optim_cfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optim_cfg.optimizer}")


def build_scheduler(optimizer, optim_cfg, epochs: int):
    if optim_cfg.scheduler == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    elif optim_cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif optim_cfg.scheduler == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {optim_cfg.scheduler}")


def run_epoch(model, loader, criterion, optimizer, device, grad_clip_norm, scaler, train: bool):
    model.train() if train else model.eval()
    total_loss, all_logits, all_labels = 0.0, [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, desc="train" if train else "eval", leave=False):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            if scaler is not None and train:
                with torch.autocast(device_type=device.type, enabled=True):
                    logits = model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                if grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                if train:
                    loss.backward()
                    if grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    avg_loss = total_loss / len(loader.dataset)
    preds = all_logits.argmax(dim=1).numpy()
    labels_np = all_labels.numpy()

    metrics = compute_core_metrics(labels_np, preds, num_classes=all_logits.shape[1])
    metrics["loss"] = avg_loss
    return metrics


def train_experiment(cfg: ExperimentConfig) -> dict:
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)
    logger.info(f"Running experiment '{cfg.name}' on {device} | git={get_git_commit_hash()}")
    logger.info(f"Hardware: {get_hardware_info()}")

    train_loader, val_loader, test_loader, class_weights, class_names = get_dataloaders(cfg)
    cfg.data.num_classes = len(class_names)
    cfg.data.class_names = class_names

    model = get_model(
        cfg.model.architecture, num_classes=cfg.data.num_classes,
        pretrained=cfg.model.pretrained, dropout=cfg.model.dropout,
    ).to(device)
    logger.info(f"Model '{cfg.model.architecture}': {count_parameters(model):,} params, {model_size_mb(model):.2f} MB")

    # Stage 1: freeze backbone, train head only.
    model.set_backbone_trainable(False)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = build_optimizer(model, cfg.optim)
    scheduler = build_scheduler(optimizer, cfg.optim, cfg.train.epochs)
    early_stopper = EarlyStopping(patience=cfg.train.early_stopping_patience, mode="max")
    scaler = torch.cuda.amp.GradScaler() if (cfg.train.mixed_precision and device.type == "cuda") else None

    run_dir = Path(cfg.train.checkpoint_dir) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(Path(cfg.train.log_dir) / cfg.name))

    start_epoch = 0
    history = []

    if cfg.train.resume_from:
        checkpoint = load_checkpoint(cfg.train.resume_from, model, optimizer, map_location=device)
        start_epoch = checkpoint["epoch"] + 1
        history = checkpoint.get("extra", {}).get("history", [])
        logger.info(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, cfg.train.epochs):
        if epoch == cfg.model.freeze_backbone_epochs:
            logger.info(f"Epoch {epoch}: unfreezing backbone for full fine-tuning")
            model.set_backbone_trainable(True)

        t0 = time.perf_counter()
        train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, device, cfg.optim.grad_clip_norm, scaler, train=True
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, optimizer, device, cfg.optim.grad_clip_norm, scaler, train=False
        )
        epoch_time = time.perf_counter() - t0

        val_score = val_metrics[cfg.train.early_stopping_metric.replace("val_", "")]

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_score)
            else:
                scheduler.step()

        writer.add_scalars("loss", {"train": train_metrics["loss"], "val": val_metrics["loss"]}, epoch)
        writer.add_scalars("f1_macro", {"train": train_metrics["f1_macro"], "val": val_metrics["f1_macro"]}, epoch)
        writer.add_scalar("lr_head", optimizer.param_groups[0]["lr"], epoch)

        is_best = early_stopper.step(val_score)

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_f1_macro": train_metrics["f1_macro"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_accuracy": val_metrics["accuracy"],
            "backbone_frozen": epoch < cfg.model.freeze_backbone_epochs,
            "epoch_time_sec": epoch_time,
        }
        history.append(record)
        logger.info(
            f"[{cfg.name}] epoch {epoch:03d} | train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_f1_macro={val_metrics['f1_macro']:.4f} "
            f"{'(best)' if is_best else ''} | {epoch_time:.1f}s"
        )

        save_checkpoint(
            str(run_dir / "last.pt"), model, optimizer, epoch, early_stopper.best_score,
            extra={"history": history, "class_names": class_names, "architecture": cfg.model.architecture},
        )
        if is_best:
            save_checkpoint(
                str(run_dir / "best.pt"), model, optimizer, epoch, early_stopper.best_score,
                extra={"history": history, "class_names": class_names, "architecture": cfg.model.architecture},
            )

        if early_stopper.should_stop:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    writer.close()

    result = {
        "experiment_name": cfg.name,
        "config": cfg.to_dict(),
        "git_commit": get_git_commit_hash(),
        "hardware": get_hardware_info(),
        "history": history,
        "best_val_f1_macro": early_stopper.best_score,
        "num_params": count_parameters(model),
        "model_size_mb": model_size_mb(model),
    }

    results_path = Path("outputs") / f"{cfg.name}_train_result.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASL Vision AI's EfficientNetV2 classifier.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = ExperimentConfig.load(args.config)
    if args.resume:
        cfg.train.resume_from = args.resume

    train_experiment(cfg)
