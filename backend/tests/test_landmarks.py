"""tests/test_landmarks.py -- app/ml/landmarks.py's parse_landmark_frame."""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.landmarks import FEATURE_DIM, InvalidLandmarkFrame, parse_landmark_frame


def test_feature_dim_matches_pose_plus_two_hands():
    # 33 pose landmarks * 4 (x,y,z,visibility) + 2 * 21 hand landmarks * 3 (x,y,z)
    assert FEATURE_DIM == 33 * 4 + 2 * 21 * 3 == 258


def test_parse_valid_frame_returns_float32_array_of_correct_length():
    features = [0.1] * FEATURE_DIM
    arr = parse_landmark_frame(features)
    assert arr.shape == (FEATURE_DIM,)
    assert arr.dtype == np.float32
    assert np.allclose(arr, 0.1)


def test_parse_rejects_short_vector():
    with pytest.raises(InvalidLandmarkFrame, match=r"258"):
        parse_landmark_frame([0.0] * 10)


def test_parse_rejects_long_vector():
    with pytest.raises(InvalidLandmarkFrame):
        parse_landmark_frame([0.0] * (FEATURE_DIM + 5))


def test_parse_rejects_nan():
    features = [0.0] * FEATURE_DIM
    features[10] = float("nan")
    with pytest.raises(InvalidLandmarkFrame, match="NaN"):
        parse_landmark_frame(features)


def test_parse_rejects_inf():
    features = [0.0] * FEATURE_DIM
    features[10] = float("inf")
    with pytest.raises(InvalidLandmarkFrame, match="NaN"):
        parse_landmark_frame(features)


def test_parse_rejects_non_numeric():
    features = [0.0] * FEATURE_DIM
    features[5] = "not-a-number"
    with pytest.raises(InvalidLandmarkFrame):
        parse_landmark_frame(features)


def test_parse_rejects_empty_list():
    with pytest.raises(InvalidLandmarkFrame):
        parse_landmark_frame([])
