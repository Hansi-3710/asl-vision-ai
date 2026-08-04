"""
model.py
========
The "AI Pipeline" step from the source design doc, as actual code:

    landmark sequence (T x FEATURE_DIM)
        -> linear input projection to d_model
        -> sinusoidal positional encoding
        -> Transformer encoder (self-attention over time)
        -> linear head to (vocab_size) logits per frame
        -> (training)  CTC loss against the target gloss sequence
        -> (inference) CTC greedy decode into a gloss sequence

Why a Transformer encoder over the doc's alternative (ST-GCN)? Both are
legitimate choices -- ST-GCN treats the skeleton as a graph and convolves
over joints+time, which is elegant for hand-shape detail but noticeably
more code (graph adjacency construction, multiple conv variants) for a
scaffold whose weights are explicitly a placeholder. A Transformer encoder
over a flattened per-frame feature vector needs no graph construction,
is standard PyTorch (`nn.TransformerEncoder`), and is exactly what the
doc's "OR" offered as the lighter-weight option. Swapping in an ST-GCN
later only touches this file and export_onnx.py -- everything downstream
(dataset, training loop, CTC decode, ONNX serving) is architecture-agnostic.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al.), added to
    the input embeddings so the encoder has a notion of frame order --
    self-attention alone is permutation-invariant and would otherwise
    treat a shuffled sequence identically to the original."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1)]


class LandmarkTransformer(nn.Module):
    """Encoder-only Transformer + CTC head over landmark-feature sequences.

    Input:  (batch, seq_len, feature_dim) float32 landmark features
    Output: (batch, seq_len, vocab_size) log-probabilities per frame,
            ready for nn.CTCLoss (which expects log-probs) or greedy/beam
            CTC decoding.
    """

    def __init__(
        self,
        feature_dim: int,
        vocab_size: int,
        d_model: int = 64,
        num_encoder_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_position_embeddings: int = 128,
    ):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_position_embeddings)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.ctc_head = nn.Linear(d_model, vocab_size)

        self.feature_dim = feature_dim
        self.vocab_size = vocab_size
        self.d_model = d_model

    def forward(self, features: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        features: (batch, seq_len, feature_dim)
        key_padding_mask: (batch, seq_len) bool, True at PADDED positions
            (matches nn.TransformerEncoderLayer's convention). Pass this
            whenever sequences in the batch are padded to different
            original lengths so attention doesn't attend to pad frames.
        Returns: (batch, seq_len, vocab_size) log-probabilities.
        """
        x = self.input_proj(features)
        x = self.pos_encoding(x)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.layer_norm(x)
        logits = self.ctc_head(x)
        return torch.log_softmax(logits, dim=-1)


def greedy_ctc_decode(log_probs: torch.Tensor, blank_id: int = 0) -> list[list[int]]:
    """Standard CTC greedy decode: argmax per frame, collapse consecutive
    repeats, then drop blanks. This is what the doc's "CTC Decoder" box
    means concretely -- no external decoder library required.

    log_probs: (batch, seq_len, vocab_size)
    Returns: list (length=batch) of decoded id sequences (blanks removed,
    repeats collapsed).
    """
    ids = log_probs.argmax(dim=-1)  # (batch, seq_len)
    decoded_batch: list[list[int]] = []
    for row in ids.tolist():
        decoded: list[int] = []
        prev = None
        for tok in row:
            if tok != prev and tok != blank_id:
                decoded.append(tok)
            prev = tok
        decoded_batch.append(decoded)
    return decoded_batch


def get_model(feature_dim: int, vocab_size: int, **kwargs) -> LandmarkTransformer:
    return LandmarkTransformer(feature_dim=feature_dim, vocab_size=vocab_size, **kwargs)


if __name__ == "__main__":
    torch.manual_seed(0)
    feature_dim, vocab_size, batch, seq_len = 258, 14, 3, 40

    model = get_model(feature_dim, vocab_size)
    x = torch.randn(batch, seq_len, feature_dim)
    log_probs = model(x)
    assert log_probs.shape == (batch, seq_len, vocab_size), log_probs.shape
    # log_softmax output: each frame's distribution should sum to ~1 in prob space.
    assert torch.allclose(log_probs.exp().sum(dim=-1), torch.ones(batch, seq_len), atol=1e-4)

    # padding mask shouldn't crash and should still produce the right shape
    mask = torch.zeros(batch, seq_len, dtype=torch.bool)
    mask[:, 30:] = True  # last 10 frames are padding
    log_probs_masked = model(x, key_padding_mask=mask)
    assert log_probs_masked.shape == (batch, seq_len, vocab_size)

    decoded = greedy_ctc_decode(log_probs)
    assert len(decoded) == batch
    assert all(isinstance(seq, list) for seq in decoded)

    # A hand-built log-prob tensor to check collapse-repeats + drop-blanks
    # logic exactly, rather than relying on random model output.
    # sequence: blank, 3, 3, blank, blank, 5, 5, 5, blank -> should decode to [3, 5]
    fake = torch.full((1, 9, vocab_size), -10.0)
    for t, tok in enumerate([0, 3, 3, 0, 0, 5, 5, 5, 0]):
        fake[0, t, tok] = 0.0
    assert greedy_ctc_decode(fake, blank_id=0) == [[3, 5]], greedy_ctc_decode(fake, blank_id=0)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model.py OK -- {n_params:,} params, output shape correct, CTC greedy decode logic correct.")
