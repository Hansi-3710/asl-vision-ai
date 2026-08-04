"""
baseline.py
===========
Trains a non-deep-learning baseline so the CNN's accuracy has a reference
point ("the model got 94%" means little without something to compare it
to). Uses HOG (Histogram of Oriented Gradients) features + a linear SVM --
a strong, still-non-deep baseline well-suited to hand-shape/edge-orientation
discrimination.

Self-sufficient w.r.t. the train/val/test split: if the split file at
`--split_path` doesn't exist yet, this creates it via the same group-aware
`stratified_split` that `dataset.get_dataloaders()` uses (so it works
whether this is run before or after `train.py` -- it doesn't depend on
call order to have already produced a split).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

from dataset import stratified_split, load_split_indices
from logger import get_logger

logger = get_logger(__name__)


def _get_or_create_split(data_dir: str, split_path: str, train_frac=0.70, val_frac=0.15, test_frac=0.15, seed=42):
    if os.path.exists(split_path):
        logger.info(f"Using existing split at {split_path}")
        return load_split_indices(split_path, data_dir)
    logger.info(f"No split found at {split_path} -- creating one now.")
    return stratified_split(data_dir, train_frac, val_frac, test_frac, seed, split_path)


def _extract_hog_features(samples: list, image_size: int = 64) -> tuple:
    X, y = [], []
    for path, label in samples:
        img = Image.open(path).convert("L").resize((image_size, image_size))
        feat = hog(
            np.asarray(img),
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
        )
        X.append(feat)
        y.append(label)
    return np.stack(X), np.array(y)


def _load_images_as_arrays(samples: list, image_size: int = 32) -> tuple:
    X, y = [], []
    for path, label in samples:
        img = Image.open(path).convert("L").resize((image_size, image_size))
        X.append(np.asarray(img, dtype=np.float32).flatten() / 255.0)
        y.append(label)
    return np.stack(X), np.array(y)


def run_baseline(
    data_dir: str, split_path: str, method: str = "hog_svm", max_train: int = 8000, max_test: int = 2000
) -> dict:
    train_samples, _val_samples, test_samples, class_to_idx = _get_or_create_split(data_dir, split_path)

    # Baselines are for a reference number, not a leaderboard entry --
    # subsample for tractable CPU runtime on the full ~87k-image dataset.
    rng = np.random.default_rng(0)
    if len(train_samples) > max_train:
        idx = rng.choice(len(train_samples), max_train, replace=False)
        train_samples = [train_samples[i] for i in idx]
    if len(test_samples) > max_test:
        idx = rng.choice(len(test_samples), max_test, replace=False)
        test_samples = [test_samples[i] for i in idx]

    logger.info(f"Baseline method={method}, train_n={len(train_samples)}, test_n={len(test_samples)}")

    start = time.perf_counter()
    if method == "logreg_pixels":
        X_train, y_train = _load_images_as_arrays(train_samples)
        X_test, y_test = _load_images_as_arrays(test_samples)
        clf = LogisticRegression(max_iter=1000)
    elif method == "hog_svm":
        X_train, y_train = _extract_hog_features(train_samples)
        X_test, y_test = _extract_hog_features(test_samples)
        clf = LinearSVC(max_iter=5000)
    else:
        raise ValueError(f"Unknown baseline method: {method}")

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    elapsed = time.perf_counter() - start

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")

    result = {
        "method": method,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "train_n": len(train_samples),
        "test_n": len(test_samples),
        "wall_clock_sec": elapsed,
        "classification_report": classification_report(y_test, preds, output_dict=True, zero_division=0),
    }
    logger.info(f"Baseline [{method}] accuracy={acc:.4f} macro_f1={macro_f1:.4f} ({elapsed:.1f}s)")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a non-deep baseline for comparison against the CNN.")
    parser.add_argument("--data_dir", default="data/asl_alphabet_train")
    parser.add_argument("--split_path", default="outputs/split_indices.json")
    parser.add_argument("--method", choices=["logreg_pixels", "hog_svm"], default="hog_svm")
    parser.add_argument("--out", default="outputs/baseline_results.json")
    args = parser.parse_args()

    result = run_baseline(args.data_dir, args.split_path, args.method)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved baseline results to {args.out}")
