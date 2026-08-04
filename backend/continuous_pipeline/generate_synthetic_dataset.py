"""
generate_synthetic_dataset.py
==============================
IMPORTANT -- READ THIS BEFORE TRUSTING ANY OUTPUT FROM THIS PROJECT'S
DEFAULT CHECKPOINT.

Training a real continuous-ASL model requires WLASL or How2Sign: tens of
thousands of labeled sign-language video clips, hosted on external
servers this sandbox cannot reach (and which take real GPU-hours to
train on regardless of where you run it). That is the "hardest part is
data, not code" caveat from the source design doc, and it doesn't go
away just because the surrounding code exists.

What THIS script does instead: synthesizes a small fake dataset --
random landmark sequences where each of the 12 placeholder glosses in
vocab.SYNTHETIC_GLOSSES is associated with a distinct, noisy numeric
"signature" pattern -- purely so that `train.py`, `dataset.py`,
`export_onnx.py`, and the FastAPI streaming endpoint can all be run
end-to-end and proven to work mechanically (data loads, loss decreases,
ONNX export matches PyTorch output, the WebSocket serves valid
responses). A model trained on this data learns to recognize ARBITRARY
NUMERIC PATTERNS, not sign language. It will not respond meaningfully to
a real webcam.

To go from "the pipeline runs" to "the pipeline recognizes ASL":
  1. Download WLASL (https://dxli94.github.io/WLASL/) or How2Sign
     (https://how2sign.github.io/) yourself -- both require accepting
     dataset-specific license terms, so this can't be automated for you.
  2. Run landmark_extraction.py over the real video files to produce the
     same cache_dir/labels.json format this script produces.
  3. Point configs/*.yaml's `data.cache_dir` / `data.labels_path` at that
     real cache, increase model size in ModelConfig (64-dim/2-layer is
     sized for a 12-word synthetic vocab, not a several-thousand-gloss
     real one), and retrain with train.py unchanged.

Usage:
    python generate_synthetic_dataset.py --out data/synthetic --num-clips 300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from landmark_spec import FEATURE_DIM
from vocab import SYNTHETIC_GLOSSES, build_synthetic_vocab

FRAMES_PER_WORD = (8, 14)  # inclusive range of frames "signed" per word, before noise
NOISE_STD = 0.05


def _word_signature(word_id: int, rng: np.random.Generator) -> np.ndarray:
    """A fixed, reproducible per-word base feature vector (seeded on the
    word id, not the rng, so the same word always has the same signature
    across the whole dataset -- rng is only used for the noise added on
    top by the caller)."""
    word_rng = np.random.default_rng(seed=1000 + word_id)
    return word_rng.uniform(-1.0, 1.0, size=FEATURE_DIM).astype(np.float32)


def _synthesize_clip(word_ids: list[int], rng: np.random.Generator) -> np.ndarray:
    frames = []
    for word_id in word_ids:
        n_frames = int(rng.integers(FRAMES_PER_WORD[0], FRAMES_PER_WORD[1] + 1))
        base = _word_signature(word_id, rng)
        for _ in range(n_frames):
            frames.append(base + rng.normal(0, NOISE_STD, size=FEATURE_DIM).astype(np.float32))
    return np.stack(frames, axis=0)  # (total_frames, FEATURE_DIM)


def generate(out_dir: str, num_clips: int, min_words: int = 1, max_words: int = 4, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    vocab = build_synthetic_vocab()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    labels: dict[str, list[str]] = {}
    for clip_idx in range(num_clips):
        n_words = int(rng.integers(min_words, max_words + 1))
        # sample real glosses (ids 2..len-1, skipping blank=0/unk=1)
        chosen_ids = rng.integers(2, len(vocab), size=n_words).tolist()
        glosses = [vocab.decode_id(i) for i in chosen_ids]

        clip_array = _synthesize_clip(chosen_ids, rng)
        clip_id = f"clip_{clip_idx:05d}"
        np.save(out_path / f"{clip_id}.npy", clip_array)
        labels[clip_id] = glosses

    with open(out_path / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)
    vocab.save(str(out_path / "vocab.json"))

    print(
        f"generate_synthetic_dataset.py OK -- wrote {num_clips} synthetic clips "
        f"+ labels.json + vocab.json ({len(vocab)} tokens incl. blank/unk) to {out_path}/"
    )
    print("Reminder: this is placeholder data for pipeline smoke-testing, NOT real ASL. See module docstring.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/synthetic")
    parser.add_argument("--num-clips", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    generate(args.out, args.num_clips, seed=args.seed)
