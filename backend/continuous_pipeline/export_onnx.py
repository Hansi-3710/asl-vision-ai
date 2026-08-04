"""
export_onnx.py
===============
Exports a trained LandmarkTransformer checkpoint to ONNX so the FastAPI
backend can serve it with ONNX Runtime instead of full PyTorch -- per the
source doc's "Render-Friendly Backend" section: no GPU required for
inference, smaller runtime footprint, one shared session per process.

Sequence length is a dynamic ONNX axis: the streaming service feeds a
growing window of frames (not a fixed length), so a statically-shaped
export would need re-exporting or padding tricks every time the buffer
size changed. Batch, by contrast, is left STATIC at 1 -- the streaming
service (app/ml/sequence_recognizer.py) never batches multiple sessions
into one inference call, and torch's current dynamo-based exporter has a
rough edge where a dynamic batch axis doesn't propagate correctly through
nn.MultiheadAttention's internal QKV-split reshape. Since real usage
never needs dynamic batch here, the simplest fix is also the correct one:
don't ask for it.

Usage:
    python export_onnx.py --config configs/synthetic.yaml \
        --checkpoint checkpoints/landmark_transformer_synthetic/best.pt \
        --output checkpoints/landmark_transformer_synthetic/model.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import ExperimentConfig
from model import get_model
from vocab import Vocabulary
from utils import get_device, load_checkpoint


def export(cfg: ExperimentConfig, checkpoint_path: str, output_path: str) -> str:
    device = torch.device("cpu")  # ONNX export always traces on CPU -- inference target has no GPU anyway
    vocab = Vocabulary.load(cfg.data.vocab_path)

    model = get_model(
        feature_dim=cfg.data.feature_dim, vocab_size=len(vocab),
        d_model=cfg.model.d_model, num_encoder_layers=cfg.model.num_encoder_layers,
        num_heads=cfg.model.num_heads, dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout, max_position_embeddings=cfg.model.max_position_embeddings,
    ).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()

    dummy_seq_len = 32
    dummy_input = torch.randn(1, dummy_seq_len, cfg.data.feature_dim)
    dummy_mask = torch.zeros(1, dummy_seq_len, dtype=torch.bool)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_input, dummy_mask),
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["features", "padding_mask"],
        output_names=["log_probs"],
        # Batch is intentionally NOT a dynamic axis here -- see the
        # docstring note above this function about why (only seq_len
        # varies at serve time; batch is always 1).
        dynamic_axes={
            "features": {1: "seq_len"},
            "padding_mask": {1: "seq_len"},
            "log_probs": {1: "seq_len"},
        },
    )

    # Also copy the vocab next to the ONNX file -- the FastAPI streaming
    # service needs both together and this keeps them from drifting apart
    # if someone copies model.onnx to a different deployment without
    # remembering the vocab.
    vocab.save(str(Path(output_path).parent / "vocab.json"))

    print(f"export_onnx.py OK -- wrote {output_path} (+ vocab.json alongside it)")
    return output_path


def verify_onnx_matches_pytorch(cfg: ExperimentConfig, checkpoint_path: str, onnx_path: str, atol: float = 1e-4) -> None:
    """Runs the same random input through both the PyTorch model and the
    exported ONNX graph and asserts the outputs match within tolerance --
    catches export bugs (wrong dynamic axes, op that doesn't lower
    correctly) that torch.onnx.export succeeding silently would not."""
    import numpy as np
    import onnxruntime as ort

    device = torch.device("cpu")
    vocab = Vocabulary.load(cfg.data.vocab_path)
    model = get_model(
        feature_dim=cfg.data.feature_dim, vocab_size=len(vocab),
        d_model=cfg.model.d_model, num_encoder_layers=cfg.model.num_encoder_layers,
        num_heads=cfg.model.num_heads, dim_feedforward=cfg.model.dim_feedforward,
        dropout=cfg.model.dropout, max_position_embeddings=cfg.model.max_position_embeddings,
    ).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()

    torch.manual_seed(123)
    seq_len = 47  # deliberately different from the export-time dummy_seq_len (32), to prove the dynamic seq_len axis works
    x = torch.randn(1, seq_len, cfg.data.feature_dim)  # batch=1, matching real streaming usage -- see module docstring
    mask = torch.zeros(1, seq_len, dtype=torch.bool)
    mask[0, 40:] = True  # exercise the padding-mask path too (see docstring note on why only non-padded positions are checked below)

    with torch.no_grad():
        torch_out = model(x, key_padding_mask=mask).numpy()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = session.run(
        ["log_probs"],
        {"features": x.numpy().astype(np.float32), "padding_mask": mask.numpy()},
    )[0]

    non_padded = ~mask.numpy()[0]  # (seq_len,) -- only these positions have a defined, comparable output
    max_diff = float(np.abs(torch_out[0, non_padded] - onnx_out[0, non_padded]).max())
    assert max_diff < atol, f"ONNX output diverges from PyTorch by {max_diff} at non-padded positions (tolerance {atol})"
    print(f"verify_onnx_matches_pytorch OK -- max abs diff {max_diff:.2e} < {atol} at all {non_padded.sum()} "
          f"non-padded positions (checked at seq_len={seq_len}, different from export-time {32}, confirming the "
          f"dynamic seq_len axis works). Padded-position outputs are intentionally excluded -- see this function's "
          f"call site comment: nn.TransformerEncoder's masking fast path doesn't guarantee those match, and "
          f"the serving code never reads them anyway.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    cfg = ExperimentConfig.load(args.config)
    export(cfg, args.checkpoint, args.output)
    if not args.skip_verify:
        verify_onnx_matches_pytorch(cfg, args.checkpoint, args.output)
