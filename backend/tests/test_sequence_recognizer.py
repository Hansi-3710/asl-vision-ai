"""tests/test_sequence_recognizer.py -- StreamingSession + CTC decode + postprocess."""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.sequence_recognizer import (
    DummySequenceBackend,
    StreamingSession,
    greedy_ctc_decode_numpy,
    postprocess_transcript,
)
from continuous_pipeline.vocab import build_synthetic_vocab  # noqa: F401 -- see conftest sys.path shim below


class FixedScriptBackend:
    """A fake SequenceModelBackend that returns a pre-scripted decoded word
    sequence on every call, regardless of input -- lets tests assert on
    StreamingSession's buffering/transcript logic without needing a real
    (trained or untrained) model in the loop."""

    def __init__(self, vocab, scripted_words: list[str]):
        self.vocab = vocab
        self.is_ready = True
        self.is_placeholder = False
        self._scripted_ids = vocab.encode_sequence(scripted_words)

    def run(self, features: np.ndarray) -> np.ndarray:
        seq_len = features.shape[0]
        vocab_size = len(self.vocab)
        # Build log-probs whose greedy decode is EXACTLY self._scripted_ids:
        # one frame of "loud" logit per target id, separated by a blank
        # frame so consecutive-repeat collapsing doesn't merge them.
        log_probs = np.full((seq_len, vocab_size), -10.0, dtype=np.float32)
        log_probs[:, self.vocab.blank_id] = 0.0  # default: blank
        t = 0
        for tok_id in self._scripted_ids:
            if t >= seq_len:
                break
            log_probs[t, self.vocab.blank_id] = -10.0
            log_probs[t, tok_id] = 0.0
            t += 2  # leave a blank frame gap between words
        return log_probs


@pytest.fixture
def vocab():
    return build_synthetic_vocab()


def _words_only(structured_words: list[dict]) -> list[str]:
    """Test helper: StreamingSession.run_inference() now returns
    [{"word": ..., "confidence": ...}, ...] rather than bare strings --
    most tests only care about which words came out, not their exact
    confidence value, so pull just the words for those assertions."""
    return [w["word"] for w in structured_words]


def test_greedy_ctc_decode_numpy_collapses_repeats_and_drops_blanks(vocab):
    seq_len, vocab_size = 9, len(vocab)
    log_probs = np.full((seq_len, vocab_size), -10.0, dtype=np.float32)
    for t, tok in enumerate([0, 3, 3, 0, 0, 5, 5, 5, 0]):
        log_probs[t, tok] = 0.0
    decoded = greedy_ctc_decode_numpy(log_probs, blank_id=0)
    assert [tok_id for tok_id, _ in decoded] == [3, 5]
    # each token was "loud" (logit 0.0 vs -10.0 for every rival) at every
    # frame in its run, so confidence should be very high (near-1.0 softmax mass)
    for _, confidence in decoded:
        assert 0.99 < confidence <= 1.0


def test_greedy_ctc_decode_numpy_empty_for_all_blank():
    log_probs = np.zeros((5, 4), dtype=np.float32)
    log_probs[:, 0] = 10.0  # blank dominates every frame
    assert greedy_ctc_decode_numpy(log_probs, blank_id=0) == []


def test_postprocess_transcript_empty():
    assert postprocess_transcript([]) == ""


def test_postprocess_transcript_capitalizes_and_punctuates():
    result = postprocess_transcript(["hello", "MY", "name"])
    assert result == "Hello My Name."


def test_postprocess_transcript_keeps_pronoun_i_uppercase():
    assert postprocess_transcript(["i", "like", "computers"]) == "I Like Computers."


def test_dummy_backend_never_crashes_and_reports_not_ready():
    backend = DummySequenceBackend()
    assert backend.is_ready is False
    log_probs = backend.run(np.zeros((10, 258), dtype=np.float32))
    assert log_probs.shape[0] == 10
    # Dummy backend should only ever predict blank -> empty decode.
    assert greedy_ctc_decode_numpy(log_probs, blank_id=0) == []


def test_streaming_session_does_not_run_inference_before_stride_reached(vocab):
    backend = FixedScriptBackend(vocab, ["HELLO"])
    session = StreamingSession(backend, inference_stride_frames=8)
    for _ in range(7):
        session.add_frame(np.zeros(258, dtype=np.float32))
    assert session.should_run_inference is False


def test_streaming_session_runs_inference_once_stride_reached_and_decodes_words(vocab):
    backend = FixedScriptBackend(vocab, ["HELLO", "MY", "NAME"])
    session = StreamingSession(backend, inference_stride_frames=8, model_input_frames_cap=96)
    for _ in range(8):
        session.add_frame(np.random.randn(258).astype(np.float32))
    assert session.should_run_inference is True

    result = session.run_inference()
    assert _words_only(result["words"]) == ["HELLO", "MY", "NAME"]
    assert all(0.0 <= w["confidence"] <= 1.0 for w in result["words"])
    assert result["transcript"] == "Hello My Name."
    assert result["buffer_frames"] == 8
    assert result["latency_ms"] >= 0.0

    # Frame counter should reset after a run, so it doesn't immediately
    # claim readiness again after just one more frame.
    assert session.should_run_inference is False


def test_streaming_session_reset_clears_buffer_and_transcript(vocab):
    backend = FixedScriptBackend(vocab, ["HELLO"])
    session = StreamingSession(backend, inference_stride_frames=4)
    for _ in range(4):
        session.add_frame(np.random.randn(258).astype(np.float32))
    session.run_inference()

    session.reset()
    assert len(session._buffer) == 0
    result = session.run_inference()
    assert result["words"] == []
    assert result["transcript"] == ""
    assert result["buffer_frames"] == 0


def test_streaming_session_long_buffer_is_subsampled_not_truncated(vocab):
    """A buffer longer than model_input_frames_cap must still run cleanly
    (via uniform subsampling), not error or silently drop the tail."""
    backend = FixedScriptBackend(vocab, ["YES"])
    session = StreamingSession(
        backend, max_buffer_frames=500, model_input_frames_cap=32, inference_stride_frames=8,
    )
    for _ in range(200):
        session.add_frame(np.random.randn(258).astype(np.float32))

    result = session.run_inference()
    assert result["buffer_frames"] == 200  # buffer itself isn't trimmed...
    assert _words_only(result["words"]) == ["YES"]  # ...but inference still runs fine on the subsampled window


def test_streaming_session_ignores_inference_when_backend_not_ready():
    backend = DummySequenceBackend()
    session = StreamingSession(backend, inference_stride_frames=4)
    for _ in range(4):
        session.add_frame(np.zeros(258, dtype=np.float32))
    # should_run_inference gates on backend.is_ready -- a not-ready backend
    # should never trigger a real inference tick, only the explicit
    # run_inference() escape hatch (used by "reset") which degrades gracefully.
    assert session.should_run_inference is False
    result = session.run_inference()
    assert result["words"] == []
