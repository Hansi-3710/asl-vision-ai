"""
tests/test_landmark_extraction.py
===================================
Tests continuous_pipeline/landmark_extraction.py's PURE flatten function
against mock landmark objects -- mirrors test_hand_detector.py's approach
(no real MediaPipe Holistic model file needed, so this runs anywhere).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_CONTINUOUS_PIPELINE_DIR = Path(__file__).resolve().parents[1] / "continuous_pipeline"
_path_str = str(_CONTINUOUS_PIPELINE_DIR)
_already_on_path = _path_str in sys.path
if not _already_on_path:
    sys.path.insert(0, _path_str)
try:
    # Scoped, not permanent: see app/ml/landmarks.py's identical comment.
    # Leaving continuous_pipeline on sys.path for the rest of the pytest
    # session would shadow ml_pipeline's own config.py for every test
    # collected afterward -- the exact bug that was already fixed once in
    # app/ml/, silently reintroduced here by skipping the same care.
    from landmark_extraction import _flatten_holistic_result  # noqa: E402
    from landmark_spec import (  # noqa: E402
        FEATURE_DIM, POSE_SLICE, LEFT_HAND_SLICE, RIGHT_HAND_SLICE,
        NUM_POSE_LANDMARKS, NUM_HAND_LANDMARKS,
    )
finally:
    if not _already_on_path and _path_str in sys.path:
        sys.path.remove(_path_str)


@dataclass
class MockPoseLandmark:
    x: float
    y: float
    z: float
    visibility: float


@dataclass
class MockHandLandmark:
    x: float
    y: float
    z: float


def _make_pose(value: float = 0.5) -> list[MockPoseLandmark]:
    return [MockPoseLandmark(value, value, value, 1.0) for _ in range(NUM_POSE_LANDMARKS)]


def _make_hand(value: float = 0.5) -> list[MockHandLandmark]:
    return [MockHandLandmark(value, value, value) for _ in range(NUM_HAND_LANDMARKS)]


def test_flatten_full_detection_has_correct_length_and_dtype():
    result = _flatten_holistic_result(_make_pose(), _make_hand(), _make_hand())
    assert result.shape == (FEATURE_DIM,)
    assert result.dtype == np.float32


def test_flatten_missing_hands_are_zero_not_dropped():
    """A frame where only the pose was detected (both hands out of frame)
    must still produce a FEATURE_DIM-length vector -- missing parts are
    zero-filled, never omitted, or the sequence would silently misalign
    frame-to-frame (see landmark_spec.py's docstring)."""
    result = _flatten_holistic_result(_make_pose(0.7), None, None)
    assert result.shape == (FEATURE_DIM,)

    pose_part = result[POSE_SLICE[0]:POSE_SLICE[1]]
    left_part = result[LEFT_HAND_SLICE[0]:LEFT_HAND_SLICE[1]]
    right_part = result[RIGHT_HAND_SLICE[0]:RIGHT_HAND_SLICE[1]]

    assert np.allclose(pose_part[0::4], 0.7)  # x values
    assert np.all(left_part == 0.0)
    assert np.all(right_part == 0.0)


def test_flatten_no_detections_at_all_is_all_zero():
    result = _flatten_holistic_result(None, None, None)
    assert result.shape == (FEATURE_DIM,)
    assert np.all(result == 0.0)


def test_flatten_pose_includes_visibility_hands_do_not():
    """Pose landmarks pack (x, y, z, visibility) = 4 values; hand
    landmarks pack (x, y, z) = 3 values -- this test pins that layout
    difference so a future refactor can't silently drop visibility or
    add it to hands without a test failing."""
    pose = [MockPoseLandmark(0.1, 0.2, 0.3, 0.9)] + _make_pose()[1:]
    left_hand = [MockHandLandmark(0.4, 0.5, 0.6)] + _make_hand()[1:]

    result = _flatten_holistic_result(pose, left_hand, None)

    assert list(result[0:4]) == [np.float32(v) for v in (0.1, 0.2, 0.3, 0.9)]
    left_start = LEFT_HAND_SLICE[0]
    assert list(result[left_start:left_start + 3]) == [np.float32(v) for v in (0.4, 0.5, 0.6)]


def test_flatten_truncates_extra_landmarks_defensively():
    """If MediaPipe ever returned more landmarks than expected for a part
    (shouldn't happen in practice, but defensive code should not crash),
    the flatten function truncates rather than erroring or overflowing
    the fixed-size output."""
    too_many_pose = _make_pose() + [MockPoseLandmark(9, 9, 9, 9)] * 5
    result = _flatten_holistic_result(too_many_pose, None, None)
    assert result.shape == (FEATURE_DIM,)
