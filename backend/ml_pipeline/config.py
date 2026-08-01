"""
config.py
=========
Central configuration for the ASL Vision AI training pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DataConfig:
    data_dir: str = "data/asl_alphabet_train"
    image_size: int = 224  # EfficientNetV2-S default input resolution
    num_classes: int = 29
    class_names: Optional[list] = None
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    split_seed: int = 42
    split_indices_path: str = "outputs/split_indices.json"
    # ImageNet stats are the RIGHT choice here (unlike the from-scratch
    # research variant of this project) because we're fine-tuning
    # ImageNet-pretrained EfficientNetV2 weights -- the backbone's early
    # layers expect inputs normalized the way they were during pretraining.
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)
    imbalance_strategy: str = "weighted_loss"


@dataclass
class ModelConfig:
    architecture: str = "efficientnet_v2_s"  # efficientnet_v2_s / m / l
    pretrained: bool = True
    dropout: float = 0.3
    freeze_backbone_epochs: int = 3  # warm up the new classifier head before unfreezing the backbone


@dataclass
class OptimConfig:
    optimizer: str = "adamw"
    lr: float = 3e-4
    backbone_lr_multiplier: float = 0.1  # backbone fine-tunes slower than the new head
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    grad_clip_norm: Optional[float] = 5.0


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 30
    early_stopping_patience: int = 6
    early_stopping_metric: str = "val_f1_macro"
    num_workers: int = 4
    seed: int = 0
    device: str = "auto"
    use_augmentation: bool = True
    log_dir: str = "outputs/tensorboard"
    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None
    mixed_precision: bool = True


@dataclass
class ExperimentConfig:
    name: str = "efficientnet_v2_s_baseline"
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

        data_dict = raw.get("data", {})
        for key in ("mean", "std"):
            if key in data_dict and data_dict[key] is not None:
                data_dict[key] = tuple(data_dict[key])

        return cls(
            name=raw.get("name", "unnamed"),
            data=DataConfig(**data_dict),
            model=ModelConfig(**raw.get("model", {})),
            optim=OptimConfig(**raw.get("optim", {})),
            train=TrainConfig(**raw.get("train", {})),
        )


if __name__ == "__main__":
    cfg = ExperimentConfig(name="sanity_check")
    cfg.save("configs/_sanity_check.yaml")
    reloaded = ExperimentConfig.load("configs/_sanity_check.yaml")
    assert reloaded == cfg, "Config did not round-trip through YAML!"
    print("config.py OK -- default config serializes and reloads correctly.")
