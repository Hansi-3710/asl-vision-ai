"""
train.py
========
Trains LandmarkTransformer with CTC loss against cached landmark
sequences. See generate_synthetic_dataset.py's docstring for what "the
default config trains on" actually means (synthetic placeholder data,
not real ASL) -- swap `data.cache_dir`/`data.labels_path` in the config
to a real WLASL/How2Sign cache to train something that recognizes actual
signs.

Usage:
    python train.py --config configs/synthetic.yaml
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from config import ExperimentConfig
from dataset import LandmarkSequenceDataset, collate_fn
from model import get_model, greedy_ctc_decode
from vocab import Vocabulary
from utils import get_device, save_checkpoint, set_seed


def _batch_word_accuracy(pred_ids: list[list[int]], target_ids: list[list[int]]) -> float:
    """Fraction of samples in the batch whose decoded gloss sequence
    exactly matches the target -- a coarse but easy-to-read training
    signal alongside CTC loss (loss going down doesn't always make
    intuitive sense to read at a glance; exact-match rate does)."""
    if not pred_ids:
        return 0.0
    correct = sum(1 for p, t in zip(pred_ids, target_ids) if p == t)
    return correct / len(pred_ids)


def _unflatten_targets(targets: torch.Tensor, target_lengths: torch.Tensor) -> list[list[int]]:
    out, offset = [], 0
    for length in target_lengths.tolist():
        out.append(targets[offset: offset + length].tolist())
        offset += length
    return out


def train(cfg: ExperimentConfig) -> str:
    set_seed(cfg.train.seed)
    device = get_device(cfg.train.device)

    vocab = Vocabulary.load(cfg.data.vocab_path)
    full_dataset = LandmarkSequenceDataset(
        cfg.data.cache_dir, cfg.data.labels_path, vocab, max_sequence_length=cfg.data.max_sequence_length
    )

    val_size = max(1, int(len(full_dataset) * cfg.data.val_split))
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(cfg.data.split_seed)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.train.num_workers, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.train.num_workers, collate_fn=collate_fn,
    )

    model = get_model(
        feature_dim=cfg.data.feature_dim,
        vocab_size=len(vocab),
        d_model=cfg.model.d_model,
        num_encoder_layers=cfg.model.num_encoder_layers,
        num_heads=cfg.model.num_heads,
        dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout,
        max_position_embeddings=cfg.model.max_position_embeddings,
    ).to(device)

    ctc_loss = nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
        if cfg.optim.scheduler == "cosine" else None
    )

    best_val_loss = float("inf")
    best_checkpoint_path = f"{cfg.train.checkpoint_dir}/best.pt"
    start_epoch = 0

    if cfg.train.resume_from:
        from utils import load_checkpoint
        checkpoint = load_checkpoint(cfg.train.resume_from, model, map_location=device)
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"Resumed from {cfg.train.resume_from} at epoch {start_epoch}")

    patience_counter = 0
    t0 = time.time()

    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            features = batch["features"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            targets = batch["targets"].to(device)
            seq_lengths = batch["seq_lengths"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            optimizer.zero_grad()
            log_probs = model(features, key_padding_mask=padding_mask)  # (B, T, V)
            # nn.CTCLoss wants (T, B, V)
            loss = ctc_loss(log_probs.transpose(0, 1), targets, seq_lengths, target_lengths)
            loss.backward()
            if cfg.optim.grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip_norm)
            optimizer.step()

            running_loss += loss.item()
            if step % max(1, cfg.train.log_every_n_steps) == 0:
                print(f"epoch {epoch:03d} step {step:03d}  loss={loss.item():.4f}")

        if scheduler:
            scheduler.step()

        # --- validation ---
        model.eval()
        val_loss_total, val_batches = 0.0, 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                padding_mask = batch["padding_mask"].to(device)
                targets = batch["targets"].to(device)
                seq_lengths = batch["seq_lengths"].to(device)
                target_lengths = batch["target_lengths"].to(device)

                log_probs = model(features, key_padding_mask=padding_mask)
                loss = ctc_loss(log_probs.transpose(0, 1), targets, seq_lengths, target_lengths)
                val_loss_total += loss.item()
                val_batches += 1

                all_preds.extend(greedy_ctc_decode(log_probs.cpu(), blank_id=vocab.blank_id))
                all_targets.extend(_unflatten_targets(targets.cpu(), target_lengths.cpu()))

        val_loss = val_loss_total / max(1, val_batches)
        acc = _batch_word_accuracy(all_preds, all_targets)
        print(f"epoch {epoch:03d} DONE  train_loss={running_loss/len(train_loader):.4f}  "
              f"val_loss={val_loss:.4f}  val_exact_match={acc:.2%}  elapsed={time.time()-t0:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                best_checkpoint_path, model, epoch,
                extra={
                    "vocab_path": cfg.data.vocab_path,
                    "feature_dim": cfg.data.feature_dim,
                    "config": cfg.to_dict(),
                    "val_loss": val_loss,
                    "val_exact_match": acc,
                },
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg.train.early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (no val_loss improvement for "
                      f"{cfg.train.early_stopping_patience} epochs).")
                break

    print(f"Training complete. Best val_loss={best_val_loss:.4f}. Checkpoint: {best_checkpoint_path}")
    return best_checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = ExperimentConfig.load(args.config)
    train(cfg)
