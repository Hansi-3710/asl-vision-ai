"""
ml/sequence_recognizer.py
==========================
The streaming half of the "Render-Friendly Backend" section of the source
design doc: loads ONE shared ONNX Runtime session at startup (no GPU, no
PyTorch needed at serve time -- training-time PyTorch and serving-time
ONNX Runtime are deliberately different code paths, matching the doc's
"PyTorch (training only)" / "ONNX Runtime for inference" split), then
each WebSocket connection gets its own lightweight StreamingSession that
holds just a landmark buffer + running transcript (a few KB, not a model
copy).

Two backend implementations, selected automatically by whether an
exported ONNX model file exists:
  - `OnnxSequenceBackend`  -- the real thing.
  - `DummySequenceBackend` -- used when no model.onnx is present yet, so
    the API and frontend can still be started, connected to, and
    exercised end-to-end (returns an always-blank prediction plus a
    warning flag) instead of the WebSocket refusing to work at all.
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

_CONTINUOUS_PIPELINE_DIR = Path(__file__).resolve().parents[2] / "continuous_pipeline"
_path_str = str(_CONTINUOUS_PIPELINE_DIR)
_already_on_path = _path_str in sys.path
if not _already_on_path:
    sys.path.insert(0, _path_str)
try:
    # See app/ml/landmarks.py's identical comment: this must NOT stay on
    # sys.path permanently, or it shadows ml_pipeline's own generically-named
    # modules (config.py, model.py, ...) for the rest of the process.
    from vocab import Vocabulary  # noqa: E402
finally:
    if not _already_on_path and _path_str in sys.path:
        sys.path.remove(_path_str)

from app.core.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)


def greedy_ctc_decode_numpy(log_probs: np.ndarray, blank_id: int = 0) -> list[tuple[int, float]]:
    """Same collapse-repeats-then-drop-blanks CTC greedy decode as
    continuous_pipeline/model.py's greedy_ctc_decode, reimplemented on
    plain numpy (so the serving path never needs torch installed) AND
    extended to track a per-word confidence: the mean argmax probability
    across the run of frames that collapsed into each decoded token. This
    is what StreamingSession surfaces as each word's `confidence` --
    matching the source doc's "Every word has confidence, timestamp,
    alternative predictions" requirement (timestamps are added by the
    caller, which knows wall-clock time; alternatives would need beam
    search instead of greedy decode and are a reasonable future
    extension, not implemented here).

    log_probs: (seq_len, vocab_size).
    Returns: list of (token_id, confidence) pairs, confidence in [0, 1].
    """
    ids = log_probs.argmax(axis=-1)
    frame_probs = np.exp(log_probs.max(axis=-1))  # per-frame probability of that frame's argmax token

    decoded: list[tuple[int, float]] = []
    prev: Optional[int] = None
    run_probs: list[float] = []

    def _flush():
        if prev is not None and prev != blank_id and run_probs:
            decoded.append((prev, float(sum(run_probs) / len(run_probs))))

    for tok, p in zip(ids.tolist(), frame_probs.tolist()):
        if tok != prev:
            _flush()
            run_probs = []
        if tok != blank_id:
            run_probs.append(p)
        prev = tok
    _flush()

    return decoded


class SequenceModelBackend:
    """Common interface both backends implement."""

    is_placeholder: bool = True
    is_ready: bool = False
    vocab: Optional[Vocabulary] = None

    def run(self, features: np.ndarray) -> np.ndarray:
        """features: (seq_len, feature_dim) -> log_probs: (seq_len, vocab_size)."""
        raise NotImplementedError


class OnnxSequenceBackend(SequenceModelBackend):
    def __init__(self, onnx_path: str, vocab_path: str, is_placeholder: bool):
        import onnxruntime as ort

        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.vocab = Vocabulary.load(vocab_path)
        self.is_placeholder = is_placeholder
        self.is_ready = True
        logger.info(f"Loaded sequence model from {onnx_path} (placeholder={is_placeholder}, "
                    f"vocab_size={len(self.vocab)})")

    def run(self, features: np.ndarray) -> np.ndarray:
        seq_len = features.shape[0]
        batch = features[np.newaxis, ...].astype(np.float32)  # (1, T, D)
        mask = np.zeros((1, seq_len), dtype=bool)  # nothing padded -- caller trims the buffer already
        outputs = self.session.run(["log_probs"], {"features": batch, "padding_mask": mask})
        return outputs[0][0]  # (T, V)


class DummySequenceBackend(SequenceModelBackend):
    """Used when no ONNX checkpoint is on disk at all (e.g. a fresh clone
    that hasn't run train.py/export_onnx.py yet). Always "detects" only
    blanks, so the API/WebSocket/frontend can be started and connected to
    without crashing -- mirrors app/ml/inference.py's
    REQUIRE_MODEL_ON_STARTUP=False philosophy for the original classifier."""

    def __init__(self, vocab: Optional[Vocabulary] = None):
        self.vocab = vocab
        self.is_placeholder = True
        self.is_ready = False

    def run(self, features: np.ndarray) -> np.ndarray:
        seq_len = features.shape[0]
        vocab_size = len(self.vocab) if self.vocab else 2
        log_probs = np.full((seq_len, vocab_size), -10.0, dtype=np.float32)
        log_probs[:, 0] = 0.0  # blank_id=0 gets all the probability mass
        return log_probs


def load_sequence_backend(
    onnx_path: str, vocab_path: str, is_placeholder: bool
) -> SequenceModelBackend:
    if Path(onnx_path).exists() and Path(vocab_path).exists():
        try:
            return OnnxSequenceBackend(onnx_path, vocab_path, is_placeholder)
        except Exception as e:
            logger.error(f"Failed to load sequence model from {onnx_path}: {e}")
            return DummySequenceBackend()

    logger.warning(
        f"No sequence model found at {onnx_path} -- /ws/stream will accept connections but "
        f"only ever report blank predictions until you run continuous_pipeline/train.py + "
        f"export_onnx.py (or point SEQUENCE_MODEL_ONNX_PATH at an existing export)."
    )
    vocab = Vocabulary.load(vocab_path) if Path(vocab_path).exists() else None
    return DummySequenceBackend(vocab)


def postprocess_transcript(words: list[str]) -> str:
    """Rule-based stand-in for the doc's "Language model post-processing"
    box (raw glosses -> readable English). A real deployment could swap
    this for an LLM call (e.g. the Anthropic API) without touching
    anything else in this module -- StreamingSession only depends on this
    function's signature (list[str] -> str), not its implementation.
    Deliberately simple for now: title-cases pronoun "I", capitalizes the
    sentence, appends a period. It does NOT insert missing articles/verbs
    ("my name john" stays "My name john.") -- that needs real language
    modeling, which is exactly the doc's caveat about this being the
    hardest, least-code part of the whole project."""
    if not words:
        return ""
    normalized = [("I" if w.upper() == "I" else w.capitalize()) for w in words]
    sentence = " ".join(normalized)
    return sentence[0].upper() + sentence[1:] + "."
class StreamingSession:
    """One per WebSocket connection. Holds the growing landmark buffer
    and the transcript decoded from it so far. Deliberately re-decodes
    the ENTIRE buffer on each inference tick rather than doing true
    incremental/chunked streaming decode (which would need careful
    handling of CTC state across chunk boundaries) -- simpler and
    correct-by-construction, at the cost of redoing work as the buffer
    grows. Documented as a known simplification worth revisiting if this
    becomes a bottleneck with a real (larger) model."""

    def __init__(
        self,
        backend: SequenceModelBackend,
        max_buffer_frames: int = 600,
        model_input_frames_cap: int = 96,
        inference_stride_frames: int = 8,
    ):
        self.backend = backend
        self.max_buffer_frames = max_buffer_frames
        self.model_input_frames_cap = model_input_frames_cap
        self.inference_stride_frames = inference_stride_frames

        self._buffer: deque[np.ndarray] = deque(maxlen=max_buffer_frames)
        self._frames_since_last_run = 0
        self._last_words: list[dict] = []  # [{"word": str, "confidence": float}, ...]
        self._last_tick = time.monotonic()

    def reset(self) -> None:
        self._buffer.clear()
        self._frames_since_last_run = 0
        self._last_words = []

    def add_frame(self, features: np.ndarray) -> None:
        self._buffer.append(features)
        self._frames_since_last_run += 1

    @property
    def should_run_inference(self) -> bool:
        return (
            self.backend.is_ready
            and len(self._buffer) > 0
            and self._frames_since_last_run >= self.inference_stride_frames
        )

    def _windowed_buffer(self) -> np.ndarray:
        buf = np.stack(list(self._buffer), axis=0)  # (T, D)
        if buf.shape[0] > self.model_input_frames_cap:
            # Uniform subsample down to the model's comfortable input length --
            # same technique dataset.py uses for over-long training clips, so
            # train-time and serve-time sequence handling stay consistent.
            keep_idx = np.linspace(0, buf.shape[0] - 1, self.model_input_frames_cap).astype(int)
            buf = buf[keep_idx]
        return buf

    def run_inference(self) -> dict:
        """Runs the model on the current buffer, updates the running
        transcript, and returns a result dict ready to become a
        `StreamUpdate` response. Always safe to call even if
        `should_run_inference` is False (e.g. on an explicit flush)."""
        start = time.perf_counter()
        now = time.monotonic()
        elapsed = now - self._last_tick
        fps = (1.0 / elapsed) if elapsed > 0 else 0.0
        self._last_tick = now
        self._frames_since_last_run = 0

        if len(self._buffer) == 0 or not self.backend.is_ready:
            return {
                "transcript": postprocess_transcript([w["word"] for w in self._last_words]),
                "words": self._last_words,
                "buffer_frames": len(self._buffer),
                "latency_ms": 0.0,
                "fps": round(fps, 1),
            }

        windowed = self._windowed_buffer()
        log_probs = self.backend.run(windowed)
        blank_id = self.backend.vocab.blank_id if self.backend.vocab else 0
        decoded = greedy_ctc_decode_numpy(log_probs, blank_id=blank_id)  # [(token_id, confidence), ...]
        words = (
            [{"word": self.backend.vocab.decode_id(tok_id), "confidence": round(conf, 3)} for tok_id, conf in decoded]
            if self.backend.vocab else []
        )
        self._last_words = words

        latency_ms = (time.perf_counter() - start) * 1000
        return {
            "transcript": postprocess_transcript([w["word"] for w in words]),
            "words": words,
            "buffer_frames": len(self._buffer),
            "latency_ms": round(latency_ms, 2),
            "fps": round(fps, 1),
        }
