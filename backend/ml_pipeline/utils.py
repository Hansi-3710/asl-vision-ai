"""
utils.py
========
Reusable helpers: seeding, device selection, checkpointing, git commit
tracking, and efficiency measurement (latency + FLOPs). Adapted from an
earlier, separately-tested research pipeline -- the seeding/checkpoint
logic here is unchanged from that validated version.
"""

from __future__ import annotations

import os
import random
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:  # pragma: no cover - environment dependent
            print(f"[utils.set_seed] Could not force full determinism: {e}")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preference: str = "auto") -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but no GPU is available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_git_commit_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def get_hardware_info() -> dict:
    return {
        "cpu": _try_get_cpu_name(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "n/a",
        "torch_version": torch.__version__,
    }


def _try_get_cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return "unknown"


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: torch.nn.Module) -> float:
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buffer_bytes) / (1024 ** 2)


def count_flops(model: torch.nn.Module, input_size: tuple) -> dict:
    try:
        from thop import profile
    except ImportError:
        return {
            "macs": None,
            "flops": None,
            "note": "thop not installed -- run `pip install thop`."
        }

    # Put the dummy tensor on the same device as the model
    device = next(model.parameters()).device
    dummy = torch.randn(1, *input_size).to(device)

    was_training = model.training
    model.eval()

    with torch.no_grad():
        macs, _ = profile(model, inputs=(dummy,), verbose=False)

    model.train(was_training)

    return {
        "macs": macs,
        "flops": macs * 2
    }


@contextmanager
def timer():
    class _T:
        elapsed = None

    t = _T()
    start = time.perf_counter()
    yield t
    t.elapsed = time.perf_counter() - start


def measure_inference_latency(
    model: torch.nn.Module, input_size: tuple, device: torch.device, n_warmup: int = 10, n_runs: int = 100
) -> dict:
    model.eval()
    dummy = torch.randn(1, *input_size, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_runs):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    return {
        "ms_per_image": (elapsed / n_runs) * 1000,
        "throughput_img_per_sec": n_runs / elapsed,
        "device": str(device),
    }


def save_checkpoint(
    path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_metric: float,
    extra: Optional[dict] = None,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "git_commit": get_git_commit_hash(),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location=None,
) -> dict:
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


class EarlyStopping:
    def __init__(self, patience: int = 6, mode: str = "max", min_delta: float = 1e-4):
        assert mode in ("max", "min")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return True

        improved = (score > self.best_score + self.min_delta) if self.mode == "max" else (
            score < self.best_score - self.min_delta
        )

        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


if __name__ == "__main__":
    set_seed(0)
    print("Hardware:", get_hardware_info())
    es = EarlyStopping(patience=2, mode="max")
    for score in [0.5, 0.6, 0.6, 0.6]:
        is_best = es.step(score)
        print(score, "best?", is_best, "should_stop?", es.should_stop)
