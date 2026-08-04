"""
landmark_spec.py
=================
The SINGLE source of truth for how a video frame's MediaPipe Holistic
output is flattened into a fixed-size feature vector. Every other module
that touches landmark features -- the extraction script, the PyTorch
Dataset, the model's input layer, the FastAPI streaming service, AND the
browser-side TypeScript extractor (frontend/lib/holistic-features.ts,
kept in sync with this file BY HAND, see the comment at the top of that
file) -- must agree on this layout. If they drift apart, the model will
silently receive garbage input instead of erroring, which is much worse
than a crash. `validate_feature_vector()` exists specifically to catch
that class of bug at the FastAPI boundary.

Layout (258-dim float32 vector per frame):

    [0    : 132]  pose:       33 landmarks x (x, y, z, visibility)
    [132  : 195]  left_hand:  21 landmarks x (x, y, z)
    [195  : 258]  right_hand: 21 landmarks x (x, y, z)

Face landmarks (468 points) are deliberately EXCLUDED from the default
feature vector. The source document this project is based on calls facial
expression "optional for facial grammar" -- true ASL grammar (yes/no
questions, topic marking, negation) does lean on eyebrows/mouth/head tilt,
but including all 468 face points would balloon the feature vector to
~1650 dims for a signal that mostly matters for a handful of grammatical
markers, not word identity. FACE_ENABLED below is a single flag to extend
the layout later without touching every other module.

Missing landmarks (e.g. a hand not visible in frame) are encoded as all
zeros for that block, never dropped -- the sequence length must stay
frame-aligned, so "hand absent" is a value, not an omission.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Landmark counts per MediaPipe Holistic output ---
NUM_POSE_LANDMARKS = 33
NUM_HAND_LANDMARKS = 21  # per hand

# --- Values per landmark ---
POSE_VALUES_PER_LANDMARK = 4  # x, y, z, visibility
HAND_VALUES_PER_LANDMARK = 3  # x, y, z (no visibility for hands)

FACE_ENABLED = False  # flip on + extend FEATURE_DIM to add facial grammar later

POSE_DIM = NUM_POSE_LANDMARKS * POSE_VALUES_PER_LANDMARK  # 132
HAND_DIM = NUM_HAND_LANDMARKS * HAND_VALUES_PER_LANDMARK  # 63

# Slice boundaries, exposed so downstream code never hardcodes offsets.
POSE_SLICE = (0, POSE_DIM)
LEFT_HAND_SLICE = (POSE_DIM, POSE_DIM + HAND_DIM)
RIGHT_HAND_SLICE = (POSE_DIM + HAND_DIM, POSE_DIM + 2 * HAND_DIM)

FEATURE_DIM = POSE_DIM + 2 * HAND_DIM  # 258


@dataclass
class FeatureVectorError(ValueError):
    message: str

    def __str__(self) -> str:
        return self.message


def validate_feature_vector(vector: "list[float] | tuple[float, ...]") -> None:
    """Raises FeatureVectorError if `vector` doesn't match FEATURE_DIM.
    Called at the WebSocket boundary (app/ml/landmarks.py) so a
    frontend/backend layout mismatch fails LOUDLY with a clear message
    instead of the model silently training/predicting on misaligned
    features."""
    if len(vector) != FEATURE_DIM:
        raise FeatureVectorError(
            f"Landmark feature vector has {len(vector)} values, expected "
            f"{FEATURE_DIM} (pose={POSE_DIM} + left_hand={HAND_DIM} + "
            f"right_hand={HAND_DIM}). This means the sender (browser "
            f"MediaPipe extraction, or a training-data cache file) is out "
            f"of sync with continuous_pipeline/landmark_spec.py."
        )


if __name__ == "__main__":
    assert FEATURE_DIM == 258, f"Unexpected FEATURE_DIM: {FEATURE_DIM}"
    assert POSE_SLICE == (0, 132)
    assert LEFT_HAND_SLICE == (132, 195)
    assert RIGHT_HAND_SLICE == (195, 258)
    validate_feature_vector([0.0] * FEATURE_DIM)
    try:
        validate_feature_vector([0.0] * 10)
        raise SystemExit("validate_feature_vector should have rejected a length-10 vector")
    except FeatureVectorError:
        pass
    print(f"landmark_spec.py OK -- FEATURE_DIM={FEATURE_DIM}, slices consistent.")
