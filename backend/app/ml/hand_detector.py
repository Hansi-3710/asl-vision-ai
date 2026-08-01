"""
ml/hand_detector.py
====================
Wraps MediaPipe's HandLandmarker (the current Tasks API -- NOT the older
`mp.solutions.hands.Hands`, which is deprecated/absent in mediapipe>=0.10)
to detect a hand in a frame and compute its bounding box. This is what
lets the live camera feature draw a bounding box around the detected hand
and -- just as importantly -- lets the backend crop to the hand region
before classification, since the training images are hand-filling,
close-up shots rather than wide scenes.

SETUP REQUIRED: MediaPipe's Tasks API loads its model from a `.task` file
that is NOT bundled with the pip package and must be downloaded once:

    mkdir -p ml_pipeline/checkpoints/mediapipe
    curl -L -o ml_pipeline/checkpoints/mediapipe/hand_landmarker.task \\
        https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

(URL current as of MediaPipe's published model zoo at the time this was
written -- if it 404s, check https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
for the current model asset link.) If this file is missing, HandDetector
raises a clear error at load time rather than failing confusingly deep
inside a request.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class BoundingBox:
    """Normalized (0-1) bounding box, fraction of frame width/height --
    resolution-independent so the frontend can draw it at any video
    display size without the backend needing to know that size."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float

    def as_dict(self) -> dict:
        return {
            "x_min": round(self.x_min, 4),
            "y_min": round(self.y_min, 4),
            "x_max": round(self.x_max, 4),
            "y_max": round(self.y_max, 4),
            "confidence": round(self.confidence, 4),
        }


def compute_bounding_box_from_landmarks(hand_landmarks, handedness_score: float, padding: float = 0.08) -> BoundingBox:
    """
    Pure function, deliberately separated from any MediaPipe object
    construction, so it can be unit-tested against plain mock landmark
    objects without needing the real (network-downloaded) model file.

    hand_landmarks: an iterable of objects with .x and .y attributes in
    [0, 1] (matches mediapipe.tasks.python.components.containers.landmark.NormalizedLandmark).
    padding: fraction of the box's own size to pad on each side, so the
    crop fed to the classifier includes a margin around the hand rather
    than cropping fingertips flush to the edge.
    """
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    width, height = x_max - x_min, y_max - y_min
    x_min -= width * padding
    x_max += width * padding
    y_min -= height * padding
    y_max += height * padding

    # Clamp to valid [0, 1] frame bounds -- padding can push the box outside
    # the frame if the hand is near an edge.
    x_min, y_min = max(0.0, x_min), max(0.0, y_min)
    x_max, y_max = min(1.0, x_max), min(1.0, y_max)

    return BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, confidence=handedness_score)


def crop_to_bounding_box(image_rgb: np.ndarray, box: BoundingBox) -> np.ndarray:
    """Crops an HWC RGB uint8 array to the pixel region described by a
    normalized BoundingBox."""
    h, w = image_rgb.shape[:2]
    x1, y1 = int(box.x_min * w), int(box.y_min * h)
    x2, y2 = int(box.x_max * w), int(box.y_max * h)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, max(x2, x1 + 1)), min(h, max(y2, y1 + 1))
    return image_rgb[y1:y2, x1:x2]


class HandDetector:
    def __init__(self, model_asset_path: str, num_hands: int = 1, min_detection_confidence: float = 0.5):
        model_path = Path(model_asset_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe hand landmarker model not found at {model_path}. "
                f"See the setup instructions in the docstring at the top of hand_detector.py."
            )

        import mediapipe as mp  # deferred import: keeps this module importable
        # (e.g. for unit-testing compute_bounding_box_from_landmarks) even
        # in environments without mediapipe installed.

        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._mp = mp

    def detect(self, image_rgb: np.ndarray) -> Optional[BoundingBox]:
        """Returns the bounding box of the highest-confidence detected
        hand, or None if no hand is detected in the frame."""
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=image_rgb)
        result = self._landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return None

        # If multiple hands were requested/detected, use the first (
        # MediaPipe orders by detection confidence).
        landmarks = result.hand_landmarks[0]
        score = result.handedness[0][0].score if result.handedness and result.handedness[0] else 0.5
        return compute_bounding_box_from_landmarks(landmarks, handedness_score=score)

    def close(self) -> None:
        self._landmarker.close()
