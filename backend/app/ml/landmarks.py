"""
ml/landmarks.py
================
Parses/validates the landmark feature vectors the browser sends over the
/ws/stream WebSocket. The actual layout (pose+hands, 258 dims) is defined
ONCE in continuous_pipeline/landmark_spec.py -- imported here the same
way ml/inference.py imports ml_pipeline's config/predict (sys.path
insert to a sibling, non-package directory), so the layout can never
silently drift between training-time code and serving-time code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_CONTINUOUS_PIPELINE_DIR = Path(__file__).resolve().parents[2] / "continuous_pipeline"
_path_str = str(_CONTINUOUS_PIPELINE_DIR)
_already_on_path = _path_str in sys.path
if not _already_on_path:
    sys.path.insert(0, _path_str)
try:
    # Imported modules land in sys.modules and stay importable afterward --
    # only the sys.path ENTRY needs to be temporary. Left permanently, it
    # would shadow ml_pipeline's own same-named modules (both directories
    # have generic filenames like config.py) for any code elsewhere in the
    # process that does `from config import ...` after this point, which is
    # exactly the bug this scoping avoids: app/ml/inference.py's deferred
    # `from config import ExperimentConfig` needs to resolve to
    # ml_pipeline/config.py, never continuous_pipeline/config.py.
    from landmark_spec import FEATURE_DIM, FeatureVectorError, validate_feature_vector  # noqa: E402
finally:
    if not _already_on_path and _path_str in sys.path:
        sys.path.remove(_path_str)


class InvalidLandmarkFrame(ValueError):
    """Raised for a malformed/mis-sized frame payload -- the WS handler
    catches this and sends a {"type": "error"} message back rather than
    tearing down the whole connection over one bad frame."""


def parse_landmark_frame(features: list) -> np.ndarray:
    """features: a JSON list of floats, as sent by the browser's
    holistic-features.ts. Returns a (FEATURE_DIM,) float32 array.
    Raises InvalidLandmarkFrame if the length doesn't match FEATURE_DIM
    (frontend/backend layout drift, corrupt payload, or a client bug)."""
    try:
        validate_feature_vector(features)
    except FeatureVectorError as e:
        raise InvalidLandmarkFrame(str(e)) from e

    try:
        arr = np.asarray(features, dtype=np.float32)
    except (TypeError, ValueError) as e:
        raise InvalidLandmarkFrame(f"features could not be converted to a numeric array: {e}") from e

    if not np.isfinite(arr).all():
        raise InvalidLandmarkFrame("features contained NaN/Inf values.")

    return arr


__all__ = ["FEATURE_DIM", "InvalidLandmarkFrame", "parse_landmark_frame"]
