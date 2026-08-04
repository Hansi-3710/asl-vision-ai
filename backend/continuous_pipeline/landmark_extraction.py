"""
landmark_extraction.py
========================
Turns real sign-language VIDEO files (WLASL/How2Sign/ASLLVD, once you've
downloaded them yourself -- see generate_synthetic_dataset.py's docstring
for why that can't be automated here) into the cached .npy landmark
sequences + labels.json format dataset.py reads, using MediaPipe's
HolisticLandmarker Tasks API (pose + both hands; NOT the older
`mp.solutions.holistic`, which is absent from mediapipe>=0.10 -- same
Tasks-API-not-solutions-API choice app/ml/hand_detector.py already made
for hand detection, extended here to the full body+hands).

SETUP REQUIRED (same pattern as hand_detector.py): the Holistic Tasks
model is a `.task` file not bundled with the pip package:

    mkdir -p checkpoints/mediapipe
    curl -L -o checkpoints/mediapipe/holistic_landmarker.task \\
        https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task

(If that 404s, check https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker
for the current asset link -- Google occasionally moves these.)

The module is split in two deliberately testable pieces:
  - `_flatten_holistic_result(...)`: PURE function, landmark objects ->
    the FEATURE_DIM-length np.ndarray from landmark_spec.py. Unit-tested
    against mock landmark objects (see ../tests/test_landmark_extraction.py),
    no model file or video needed.
  - `extract_video_landmarks(...)`: the actual MediaPipe + OpenCV video
    loop, calling the pure function per frame. Requires the real model
    file and an actual video, so it's exercised by the CLI path, not unit
    tests.

Usage (once you have real videos + a WLASL-style label json mapping
video filename -> list of glosses):
    python landmark_extraction.py --videos-dir data/wlasl_raw/videos \\
        --labels data/wlasl_raw/wlasl_labels.json \\
        --out data/wlasl_cache
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from landmark_spec import (
    FEATURE_DIM, HAND_DIM, POSE_DIM, NUM_POSE_LANDMARKS, NUM_HAND_LANDMARKS,
)

DEFAULT_MODEL_PATH = "checkpoints/mediapipe/holistic_landmarker.task"


def _landmarks_to_flat(landmarks, values_per_landmark: int, expected_count: int) -> np.ndarray:
    """landmarks: a list of objects with .x/.y/.z (and .visibility for
    pose) attributes, OR an empty list/None if that body part wasn't
    detected in this frame. Always returns a fixed-length zero-padded
    array so a missing hand/pose doesn't shift the feature layout for the
    rest of the frame (see landmark_spec.py's docstring)."""
    flat = np.zeros(expected_count * values_per_landmark, dtype=np.float32)
    if not landmarks:
        return flat
    for i, lm in enumerate(landmarks[:expected_count]):
        base = i * values_per_landmark
        flat[base] = lm.x
        flat[base + 1] = lm.y
        flat[base + 2] = lm.z
        if values_per_landmark == 4:
            flat[base + 3] = getattr(lm, "visibility", 0.0)
    return flat


def _flatten_holistic_result(pose_landmarks, left_hand_landmarks, right_hand_landmarks) -> np.ndarray:
    """PURE function: three landmark lists (any may be empty/None) ->
    the FEATURE_DIM-length feature vector defined by landmark_spec.py.
    Kept free of any MediaPipe object construction so it's testable with
    plain mock objects."""
    pose_flat = _landmarks_to_flat(pose_landmarks, values_per_landmark=4, expected_count=NUM_POSE_LANDMARKS)
    left_flat = _landmarks_to_flat(left_hand_landmarks, values_per_landmark=3, expected_count=NUM_HAND_LANDMARKS)
    right_flat = _landmarks_to_flat(right_hand_landmarks, values_per_landmark=3, expected_count=NUM_HAND_LANDMARKS)

    assert pose_flat.shape[0] == POSE_DIM
    assert left_flat.shape[0] == HAND_DIM
    assert right_flat.shape[0] == HAND_DIM

    return np.concatenate([pose_flat, left_flat, right_flat]).astype(np.float32)


class HolisticLandmarkExtractor:
    """Wraps mediapipe.tasks.vision.HolisticLandmarker, running in VIDEO
    mode (frame-sequential, timestamp-aware -- distinct from the
    single-shot IMAGE mode HandDetector uses for one-off frames)."""

    def __init__(self, model_asset_path: str = DEFAULT_MODEL_PATH):
        model_path = Path(model_asset_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe Holistic landmarker model not found at {model_path}. "
                f"See the setup instructions in this module's docstring."
            )

        import mediapipe as mp  # deferred: keeps this module importable without mediapipe for unit tests

        base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.HolisticLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
        )
        self._landmarker = mp.tasks.vision.HolisticLandmarker.create_from_options(options)
        self._mp = mp

    def process_frame(self, frame_rgb: np.ndarray, timestamp_ms: int) -> np.ndarray:
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        return _flatten_holistic_result(
            getattr(result, "pose_landmarks", None),
            getattr(result, "left_hand_landmarks", None),
            getattr(result, "right_hand_landmarks", None),
        )

    def close(self) -> None:
        self._landmarker.close()


def extract_video_landmarks(video_path: str, extractor: HolisticLandmarkExtractor, sample_fps: Optional[float] = None) -> np.ndarray:
    """Reads a video file with OpenCV, runs the Holistic extractor over
    every frame (or, if sample_fps is set, a subsampled rate -- useful
    for long clips where the source frame rate is higher than the model
    needs), and returns a (T, FEATURE_DIM) array."""
    import cv2  # deferred: same reasoning as the mediapipe import above

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_stride = max(1, round(native_fps / sample_fps)) if sample_fps else 1

    frames = []
    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx % frame_stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            timestamp_ms = int((frame_idx / native_fps) * 1000)
            frames.append(extractor.process_frame(frame_rgb, timestamp_ms))
        frame_idx += 1
    cap.release()

    if not frames:
        raise ValueError(f"No frames extracted from {video_path} -- corrupt file or unreadable codec.")
    return np.stack(frames, axis=0)


def run(videos_dir: str, labels_path: str, out_dir: str, model_asset_path: str, sample_fps: Optional[float]) -> None:
    with open(labels_path, "r") as f:
        raw_labels: dict[str, list[str]] = json.load(f)  # {"video_filename.mp4": ["GLOSS1", "GLOSS2"]}

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    extractor = HolisticLandmarkExtractor(model_asset_path)

    cache_labels: dict[str, list[str]] = {}
    all_glosses: set[str] = set()
    n_ok, n_failed = 0, 0

    try:
        for filename, glosses in raw_labels.items():
            clip_id = Path(filename).stem
            video_path = Path(videos_dir) / filename
            try:
                landmarks = extract_video_landmarks(str(video_path), extractor, sample_fps=sample_fps)
            except (FileNotFoundError, ValueError) as e:
                print(f"[skip] {filename}: {e}")
                n_failed += 1
                continue

            np.save(out_path / f"{clip_id}.npy", landmarks)
            cache_labels[clip_id] = glosses
            all_glosses.update(g.upper() for g in glosses)
            n_ok += 1
    finally:
        extractor.close()

    with open(out_path / "labels.json", "w") as f:
        json.dump(cache_labels, f, indent=2)

    from vocab import Vocabulary
    Vocabulary(sorted(all_glosses)).save(str(out_path / "vocab.json"))

    print(f"landmark_extraction.py OK -- {n_ok} clips extracted, {n_failed} skipped, "
          f"{len(all_glosses)} unique glosses. Cache written to {out_path}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", required=True)
    parser.add_argument("--labels", required=True, help="JSON: {video_filename: [gloss, ...]}")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sample-fps", type=float, default=None)
    args = parser.parse_args()
    run(args.videos_dir, args.labels, args.out, args.model, args.sample_fps)
