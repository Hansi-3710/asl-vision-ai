"""
transforms.py
=============
Albumentations-based image transforms. Augmentation is applied to the
TRAINING split only -- validation/test transforms only resize + normalize,
so reported metrics reflect real-world (non-augmented) performance.

NOTE ON VERSION PINNING: Albumentations has changed a couple of transform
signatures across versions (notably `RandomResizedCrop`, which in newer
releases takes `size=(h, w)` instead of separate `height`/`width` args).
This file targets the version pinned in requirements.txt
(albumentations==1.4.7). If you upgrade albumentations, check that
transform's signature first -- it's the most likely breakage point.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int, mean: tuple, std: tuple) -> A.Compose:
    """
    Note on HorizontalFlip: it's included but disabled (p=0.0) by default.
    Flipping mirrors handedness, which could plausibly help the model
    generalize across left- and right-handed signers -- but whether any
    letters in this specific alphabet become ambiguous or visually similar
    to a different letter when mirrored isn't something to assert without
    checking (this isn't simply "b/d" style chirality, since ASL handshapes
    aren't derived from printed letterforms). Recommendation: verify with a
    few example images/an ASL reference before enabling this, and note
    whichever choice you make -- and why -- in the model card/report rather
    than leaving it as an unstated default.
    """
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Rotate(limit=15, p=0.5, border_mode=0),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0, rotate_limit=0, p=0.5, border_mode=0),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.2, p=0.5),
            A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.85, 1.0), p=0.5),
            A.HorizontalFlip(p=0.0),  # OFF by default -- see note below
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_eval_transforms(image_size: int, mean: tuple, std: tuple) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def get_inference_transform(image_size: int, mean: tuple, std: tuple) -> A.Compose:
    """Alias of get_eval_transforms, used by the FastAPI backend for
    single-image/webcam-frame inference -- kept as a separate named
    function so the API layer's intent is clear at the call site."""
    return get_eval_transforms(image_size, mean, std)
