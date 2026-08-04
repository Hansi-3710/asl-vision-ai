"""
logger.py
=========
Single place that configures logging for the whole project so every module
does `from src.logger import get_logger` and gets consistent, timestamped,
file + console output instead of scattered `print()` calls.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


_CONFIGURED = False


def get_logger(name: str, log_dir: str = "outputs/logs") -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)

    if not _CONFIGURED:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        root = logging.getLogger()
        root.setLevel(logging.INFO)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(fmt)
        root.addHandler(console_handler)

        file_handler = logging.FileHandler(Path(log_dir) / "handspeak.log")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)

        _CONFIGURED = True

    return logger
