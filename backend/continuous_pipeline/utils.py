"""utils.py -- device selection + checkpoint save/load, mirroring ../ml_pipeline/utils.py's conventions."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(preference: str = "auto") -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but no GPU is available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(path: str, model: torch.nn.Module, epoch: int, extra: Optional[dict] = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": epoch, "extra": extra or {}},
        path,
    )


def load_checkpoint(path: str, model: torch.nn.Module, map_location=None) -> dict:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
