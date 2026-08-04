"""
tests/test_hand_detector.py
============================
Tests the PURE bounding-box math (padding, clamping, cropping) against
mock landmark objects -- deliberately does NOT require the real MediaPipe
model file (which must be downloaded separately, see hand_detector.py's
docstring), so these run in any environment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.ml.hand_detector import BoundingBox, compute_bounding_box_from_landmarks, crop_to_bounding_box


@dataclass
class MockLandmark:
    x: float
    y: float


def test_bounding_box_covers_all_landmarks():
    landmarks = [MockLandmark(0.4, 0.3), MockLandmark(0.6, 0.3), MockLandmark(0.5, 0.7)]
    box = compute_bounding_box_from_landmarks(landmarks, handedness_score=0.9, padding=0.0)
    assert box.x_min == 0.4
    assert box.x_max == 0.6
    assert box.y_min == 0.3
    assert box.y_max == 0.7


def test_padding_expands_box_outward():
    landmarks = [MockLandmark(0.4, 0.3), MockLandmark(0.6, 0.7)]
    box = compute_bounding_box_from_landmarks(landmarks, handedness_score=0.9, padding=0.1)
    assert box.x_min < 0.4
    assert box.x_max > 0.6
    assert box.y_min < 0.3
    assert box.y_max > 0.7


def test_box_near_top_left_edge_clamps_to_zero():
    # width/height = 0.04; padding=1.0 -> pads by 0.04 each side -> pre-clamp x_min = 0.01-0.04 = -0.03
    landmarks = [MockLandmark(0.01, 0.01), MockLandmark(0.05, 0.05)]
    box = compute_bounding_box_from_landmarks(landmarks, handedness_score=0.8, padding=1.0)
    assert box.x_min == 0.0
    assert box.y_min == 0.0


def test_box_near_bottom_right_edge_clamps_to_one():
    landmarks = [MockLandmark(0.95, 0.95), MockLandmark(0.99, 0.99)]
    box = compute_bounding_box_from_landmarks(landmarks, handedness_score=0.8, padding=1.0)
    assert box.x_max == 1.0
    assert box.y_max == 1.0


def test_crop_matches_pixel_region():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    box = BoundingBox(x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.75, confidence=0.9)
    cropped = crop_to_bounding_box(frame, box)
    assert cropped.shape == (240, 320, 3)


def test_crop_never_produces_zero_sized_output():
    """A degenerate (zero-width/height) box shouldn't crash downstream
    processing -- crop_to_bounding_box guarantees at least a 1px region."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    box = compute_bounding_box_from_landmarks(
        [MockLandmark(0.5, 0.3), MockLandmark(0.5, 0.5)], handedness_score=0.7, padding=0.1
    )
    cropped = crop_to_bounding_box(frame, box)
    assert cropped.shape[0] > 0
    assert cropped.shape[1] > 0
