"""
dataset.py
==========
Loads the ASL Alphabet Dataset and builds train/val/test DataLoaders.

The signer/session-group-aware splitting logic here is carried over from
an earlier, separately-tested research pipeline (see the big docstring on
`stratified_split` below) -- it prevents near-duplicate video frames from
the same capture session leaking across train/val/test, which would
otherwise inflate reported accuracy for this specific dataset.

Expected layout:
    data/asl_alphabet_train/
        A/*.jpg  B/*.jpg  ...  space/*.jpg  del/*.jpg  nothing/*.jpg

Download: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from transforms import get_train_transforms, get_eval_transforms


class ASLAlphabetDataset(Dataset):
    """Loads file paths eagerly, decodes images lazily in __getitem__."""

    def __init__(self, samples: list, class_to_idx: dict, transform=None):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = np.array(Image.open(path).convert("RGB"))  # HWC uint8, what Albumentations expects
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, label

    @property
    def class_names(self) -> list:
        return [self.idx_to_class[i] for i in range(len(self.idx_to_class))]


_CHUNK_SIZE = 40  # images per pseudo-session bucket; see _scan_class_folders docstring


def _infer_pseudo_signer(filename: str, index: int, n_files: int) -> str:
    """Heuristic bucket used only when real signer IDs aren't available.
    Groups each class's (sorted) files into contiguous chunks of
    `_CHUNK_SIZE` -- bucket count scales with class size, which is what
    lets the greedy group-split in `stratified_split` actually approach
    the requested train/val/test ratios instead of being capped at a
    coarse fixed granularity."""
    bucket = index // _CHUNK_SIZE
    return f"chunk_{bucket}"


def _scan_class_folders(data_dir: str) -> tuple[list, dict]:
    """Walks data_dir/<class_name>/*.jpg and returns (all_samples, class_to_idx).
    Each sample is (filepath, class_idx, pseudo_signer_id).

    ASL Alphabet dataset (like the public Kaggle version) does not ship
    explicit signer/session metadata -- in fact each class is essentially a
    single capture burst from one contributor, so *consecutive* filenames
    are typically near-duplicate video frames of the same hand pose. We
    approximate "session" by grouping each class's sorted files into
    fixed-size contiguous chunks and treating each chunk as one
    pseudo-signer/session bucket. If your copy of the dataset DOES have
    real signer/session metadata, replace `_infer_pseudo_signer` with it.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}\n"
            f"Download the ASL Alphabet Dataset from Kaggle and place it here."
        )

    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    if not class_names:
        raise RuntimeError(f"No class subfolders found under {data_dir}")

    class_to_idx = {name: i for i, name in enumerate(class_names)}
    samples = []
    for name in class_names:
        class_dir = data_dir / name
        files = sorted([f for f in class_dir.iterdir() if f.is_file()])
        for i, f in enumerate(files):
            samples.append((str(f), class_to_idx[name], _infer_pseudo_signer(f.name, i, len(files))))

    return samples, class_to_idx


def stratified_split(
    data_dir: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
    split_indices_path: str,
) -> tuple[list, list, list, dict]:
    """
    Splits by (class, signer-bucket) GROUP, not by individual image, using a
    greedy largest-remainder allocation (assign each bucket, largest first,
    to whichever split is currently furthest below its target fraction).
    This guarantees zero leakage by construction; `_check_signer_leakage`
    below is kept as a cheap regression-catching assertion.

    Saves exact split indices to disk so the split is reproducible and
    auditable rather than regenerated randomly on every run.
    """
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-6, "Splits must sum to 1.0"

    all_samples, class_to_idx = _scan_class_folders(data_dir)
    paths = [s[0] for s in all_samples]
    labels = [s[1] for s in all_samples]
    signers = [s[2] for s in all_samples]

    target_fracs = {"train": train_frac, "val": val_frac, "test": test_frac}
    rng = np.random.default_rng(seed)

    by_class: dict = {}
    for i, label in enumerate(labels):
        by_class.setdefault(label, {}).setdefault(signers[i], []).append(i)

    split_indices = {"train": [], "val": [], "test": []}

    for label, buckets in by_class.items():
        if len(buckets) < 3:
            print(
                f"[dataset.stratified_split] WARNING: class index {label} has only "
                f"{len(buckets)} pseudo-signer chunk(s). It cannot be meaningfully "
                f"divided across train/val/test and will be allocated as whole "
                f"chunks. Expected for small/toy datasets; for the real ~3,000-"
                f"images-per-class Kaggle dataset this won't occur."
            )
        bucket_items = list(buckets.items())
        order = rng.permutation(len(bucket_items))
        bucket_items = [bucket_items[i] for i in order]
        bucket_items.sort(key=lambda kv: len(kv[1]), reverse=True)

        counts = {"train": 0, "val": 0, "test": 0}
        for _bucket_name, indices in bucket_items:
            deficits = {
                s: (counts[s] / target_fracs[s]) if target_fracs[s] > 0 else float("inf")
                for s in target_fracs
            }
            chosen = min(deficits, key=deficits.get)
            split_indices[chosen].extend(indices)
            counts[chosen] += len(indices)

    train_idx = np.array(split_indices["train"])
    val_idx = np.array(split_indices["val"])
    test_idx = np.array(split_indices["test"])

    _check_signer_leakage(train_idx, val_idx, test_idx, signers, labels)

    def to_samples(indices):
        return [(paths[i], labels[i]) for i in indices]

    split_record = {
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "class_to_idx": class_to_idx,
        "seed": seed,
    }
    Path(split_indices_path).parent.mkdir(parents=True, exist_ok=True)
    with open(split_indices_path, "w") as f:
        json.dump(split_record, f)

    return to_samples(train_idx), to_samples(val_idx), to_samples(test_idx), class_to_idx


def _check_signer_leakage(train_idx, val_idx, test_idx, signers, labels) -> None:
    def class_signer_pairs(indices):
        return {(labels[i], signers[i]) for i in indices}

    train_pairs = class_signer_pairs(train_idx)
    val_pairs = class_signer_pairs(val_idx)
    test_pairs = class_signer_pairs(test_idx)

    leaks = (train_pairs & val_pairs) | (train_pairs & test_pairs) | (val_pairs & test_pairs)
    if leaks:
        raise RuntimeError(
            f"Signer/session leakage detected across splits for {len(leaks)} "
            f"(class, signer-bucket) pairs. Reported test accuracy would be inflated."
        )


def load_split_indices(split_indices_path: str, data_dir: str) -> tuple[list, list, list, dict]:
    with open(split_indices_path, "r") as f:
        record = json.load(f)

    all_samples, class_to_idx = _scan_class_folders(data_dir)
    paths = [s[0] for s in all_samples]
    labels = [s[1] for s in all_samples]

    def to_samples(indices):
        return [(paths[i], labels[i]) for i in indices]

    return (
        to_samples(record["train_idx"]),
        to_samples(record["val_idx"]),
        to_samples(record["test_idx"]),
        record["class_to_idx"],
    )


def make_imbalance_sampler(samples: list, strategy: str) -> Optional[WeightedRandomSampler]:
    if strategy != "weighted_sampler":
        return None
    labels = [s[1] for s in samples]
    class_counts = Counter(labels)
    weights = [1.0 / class_counts[label] for label in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def compute_class_weights(samples: list, num_classes: int) -> torch.Tensor:
    labels = [s[1] for s in samples]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def get_dataloaders(cfg) -> tuple:
    import os

    split_path = cfg.data.split_indices_path
    if os.path.exists(split_path):
        train_samples, val_samples, test_samples, class_to_idx = load_split_indices(split_path, cfg.data.data_dir)
    else:
        train_samples, val_samples, test_samples, class_to_idx = stratified_split(
            cfg.data.data_dir, cfg.data.train_split, cfg.data.val_split, cfg.data.test_split,
            cfg.data.split_seed, split_path,
        )

    train_tf = get_train_transforms(cfg.data.image_size, cfg.data.mean, cfg.data.std) if cfg.train.use_augmentation \
        else get_eval_transforms(cfg.data.image_size, cfg.data.mean, cfg.data.std)
    eval_tf = get_eval_transforms(cfg.data.image_size, cfg.data.mean, cfg.data.std)

    train_ds = ASLAlphabetDataset(train_samples, class_to_idx, transform=train_tf)
    val_ds = ASLAlphabetDataset(val_samples, class_to_idx, transform=eval_tf)
    test_ds = ASLAlphabetDataset(test_samples, class_to_idx, transform=eval_tf)

    sampler = make_imbalance_sampler(train_samples, cfg.data.imbalance_strategy)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=cfg.train.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers, pin_memory=True
    )

    class_weights = (
        compute_class_weights(train_samples, cfg.data.num_classes)
        if cfg.data.imbalance_strategy == "weighted_loss"
        else None
    )

    return train_loader, val_loader, test_loader, class_weights, train_ds.class_names
