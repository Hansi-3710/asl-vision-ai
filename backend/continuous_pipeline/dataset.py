"""
dataset.py
==========
Loads landmark sequences cached by generate_synthetic_dataset.py (or, for
a real run, landmark_extraction.py) as (features, target_ids) pairs for
CTC training, with a collate function that pads variable-length
sequences/labels within a batch and returns the length tensors
nn.CTCLoss requires.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from landmark_spec import FEATURE_DIM, validate_feature_vector
from vocab import Vocabulary


class LandmarkSequenceDataset(Dataset):
    def __init__(self, cache_dir: str, labels_path: str, vocab: Vocabulary, max_sequence_length: int = 64):
        self.cache_dir = Path(cache_dir)
        with open(labels_path, "r") as f:
            self.labels: dict[str, list[str]] = json.load(f)
        self.clip_ids = sorted(self.labels.keys())
        self.vocab = vocab
        self.max_sequence_length = max_sequence_length

        if not self.clip_ids:
            raise ValueError(f"No clips found in {labels_path} -- did you run generate_synthetic_dataset.py "
                              f"or landmark_extraction.py first?")

    def __len__(self) -> int:
        return len(self.clip_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        clip_id = self.clip_ids[idx]
        features = np.load(self.cache_dir / f"{clip_id}.npy")  # (T, FEATURE_DIM)
        validate_feature_vector(features[0])

        if features.shape[0] > self.max_sequence_length:
            # Uniformly subsample rather than truncate from the start, so a
            # long clip doesn't just lose everything after its first few
            # words -- CTC only needs the RELATIVE order of frames preserved.
            keep_idx = np.linspace(0, features.shape[0] - 1, self.max_sequence_length).astype(int)
            features = features[keep_idx]

        target_ids = self.vocab.encode_sequence(self.labels[clip_id])

        return torch.from_numpy(features.astype(np.float32)), torch.tensor(target_ids, dtype=torch.long)


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]):
    """Pads features to the batch's longest sequence and concatenates
    targets (CTCLoss wants targets as one flat 1D tensor + a length per
    sample, not a padded 2D tensor -- padding targets would need a
    padding-id that isn't in the vocab, adding complexity for no benefit
    since CTCLoss already handles variable target length natively)."""
    features, targets = zip(*batch)

    seq_lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)
    target_lengths = torch.tensor([t.shape[0] for t in targets], dtype=torch.long)

    max_len = int(seq_lengths.max())
    padded_features = torch.zeros(len(features), max_len, FEATURE_DIM, dtype=torch.float32)
    padding_mask = torch.ones(len(features), max_len, dtype=torch.bool)  # True = padded
    for i, f in enumerate(features):
        padded_features[i, : f.shape[0]] = f
        padding_mask[i, : f.shape[0]] = False

    flat_targets = torch.cat(targets)

    return {
        "features": padded_features,
        "padding_mask": padding_mask,
        "seq_lengths": seq_lengths,
        "targets": flat_targets,
        "target_lengths": target_lengths,
    }


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from vocab import build_synthetic_vocab

    vocab = build_synthetic_vocab()
    ds = LandmarkSequenceDataset("data/synthetic", "data/synthetic/labels.json", vocab, max_sequence_length=64)
    assert len(ds) > 0

    features, targets = ds[0]
    assert features.shape[1] == FEATURE_DIM
    assert targets.ndim == 1 and targets.numel() > 0

    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))
    assert batch["features"].shape == (8, batch["seq_lengths"].max().item(), FEATURE_DIM)
    assert batch["padding_mask"].shape == batch["features"].shape[:2]
    assert batch["targets"].numel() == batch["target_lengths"].sum().item()
    # CTC requires input_length >= target_length for every sample in the batch,
    # or the loss is undefined for that sample -- verify our synthetic data
    # (>=8 frames/word) satisfies this given max_sequence_length subsampling.
    assert (batch["seq_lengths"] >= batch["target_lengths"]).all(), \
        "Some sample has more target tokens than input frames -- CTC loss would be invalid for it."

    print(f"dataset.py OK -- {len(ds)} clips, batch shapes correct, CTC length constraint satisfied.")
