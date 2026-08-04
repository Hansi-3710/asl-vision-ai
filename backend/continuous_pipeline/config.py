"""
config.py
=========
Central configuration for the continuous ASL recognition pipeline
(landmarks -> transformer encoder -> CTC decoder -> gloss sequence).
Mirrors the structure of ../ml_pipeline/config.py deliberately, so anyone
who has read that pipeline already recognizes this one.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml

from landmark_spec import FEATURE_DIM


@dataclass
class DataConfig:
    # Directory of cached landmark sequences produced by
    # landmark_extraction.py -- one .npy file per video/clip, plus a
    # labels.json mapping clip_id -> list[str] of glosses signed in it.
    cache_dir: str = "data/synthetic"
    labels_path: str = "data/synthetic/labels.json"
    vocab_path: str = "data/synthetic/vocab.json"
    feature_dim: int = FEATURE_DIM
    # Sequences are padded/truncated to this many frames for batching.
    # 64 frames at a nominal ~15fps client sampling rate is a little
    # over 4 seconds -- enough for a short phrase, matching the "32-64
    # frame sequence buffer" called for in the source design doc.
    max_sequence_length: int = 64
    train_split: float = 0.80
    val_split: float = 0.20
    split_seed: int = 42


@dataclass
class ModelConfig:
    architecture: str = "landmark_transformer"
    d_model: int = 64
    num_encoder_layers: int = 2
    num_heads: int = 4
    dim_feedforward: int = 128
    dropout: float = 0.1
    max_position_embeddings: int = 128


@dataclass
class OptimConfig:
    optimizer: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    grad_clip_norm: Optional[float] = 5.0


@dataclass
class TrainConfig:
    batch_size: int = 16
    epochs: int = 15
    early_stopping_patience: int = 5
    num_workers: int = 0  # 0 by default: cached .npy sequences are tiny, workers add overhead not throughput
    seed: int = 0
    device: str = "auto"
    log_every_n_steps: int = 5
    checkpoint_dir: str = "checkpoints/landmark_transformer_synthetic"
    resume_from: Optional[str] = None


@dataclass
class ExperimentConfig:
    name: str = "landmark_transformer_synthetic"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(
            name=raw.get("name", "unnamed"),
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            optim=OptimConfig(**raw.get("optim", {})),
            train=TrainConfig(**raw.get("train", {})),
        )


if __name__ == "__main__":
    cfg = ExperimentConfig(name="sanity_check")
    cfg.save("configs/_sanity_check.yaml")
    reloaded = ExperimentConfig.load("configs/_sanity_check.yaml")
    assert reloaded == cfg, "Config did not round-trip through YAML!"
    Path("configs/_sanity_check.yaml").unlink()
    print("config.py OK -- default config serializes and reloads correctly.")
