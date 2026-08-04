"""
evaluate.py
===========
Computes Word Error Rate (WER) -- the standard metric for continuous sign
language recognition (edit distance between predicted and target gloss
sequences, divided by target length) -- over a held-out split, plus
exact-sequence accuracy for an easier-to-read secondary number.

Usage:
    python evaluate.py --config configs/synthetic.yaml \
        --checkpoint checkpoints/landmark_transformer_synthetic/best.pt
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader, random_split

from config import ExperimentConfig
from dataset import LandmarkSequenceDataset, collate_fn
from model import get_model, greedy_ctc_decode
from vocab import Vocabulary
from utils import get_device, load_checkpoint


def word_error_rate(pred: list[int], target: list[int]) -> float:
    """Levenshtein edit distance between two id sequences, normalized by
    target length (standard WER definition). Returns 0.0 for an empty
    target with an empty prediction (nothing to get wrong), 1.0 for an
    empty target with a non-empty prediction (all insertions)."""
    n, m = len(target), len(pred)
    if n == 0:
        return 0.0 if m == 0 else 1.0

    # classic DP edit-distance table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if target[i - 1] == pred[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution/match
            )
    return dp[n][m] / n


@torch.no_grad()
def evaluate(cfg: ExperimentConfig, checkpoint_path: str) -> dict:
    device = get_device(cfg.train.device)
    vocab = Vocabulary.load(cfg.data.vocab_path)

    full_dataset = LandmarkSequenceDataset(
        cfg.data.cache_dir, cfg.data.labels_path, vocab, max_sequence_length=cfg.data.max_sequence_length
    )
    val_size = max(1, int(len(full_dataset) * cfg.data.val_split))
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(cfg.data.split_seed)
    _, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=collate_fn)

    model = get_model(
        feature_dim=cfg.data.feature_dim, vocab_size=len(vocab),
        d_model=cfg.model.d_model, num_encoder_layers=cfg.model.num_encoder_layers,
        num_heads=cfg.model.num_heads, dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout, max_position_embeddings=cfg.model.max_position_embeddings,
    ).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()

    total_wer, exact_matches, n_samples = 0.0, 0, 0
    for batch in val_loader:
        features = batch["features"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        target_lengths = batch["target_lengths"].tolist()
        targets = batch["targets"].tolist()

        log_probs = model(features, key_padding_mask=padding_mask)
        preds = greedy_ctc_decode(log_probs.cpu(), blank_id=vocab.blank_id)

        offset = 0
        for pred, length in zip(preds, target_lengths):
            target = targets[offset: offset + length]
            offset += length
            total_wer += word_error_rate(pred, target)
            exact_matches += int(pred == target)
            n_samples += 1

    results = {
        "n_samples": n_samples,
        "word_error_rate": total_wer / max(1, n_samples),
        "exact_match_accuracy": exact_matches / max(1, n_samples),
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    cfg = ExperimentConfig.load(args.config)
    results = evaluate(cfg, args.checkpoint)
    print(f"n_samples={results['n_samples']}  WER={results['word_error_rate']:.2%}  "
          f"exact_match={results['exact_match_accuracy']:.2%}")
