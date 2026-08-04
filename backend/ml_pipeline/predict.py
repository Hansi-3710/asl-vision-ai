"""
predict.py
==========
Loads a trained checkpoint and runs inference on a single image (PIL
Image, numpy array, or raw file bytes). This is the module the FastAPI
backend imports directly for both the /predict (upload) and
/predict-webcam (frame) endpoints -- both ultimately call
`Predictor.predict`.

Usage (CLI):
    python predict.py --config configs/baseline.yaml \
        --checkpoint checkpoints/efficientnet_v2_s_baseline/best.pt \
        --image path/to/hand.jpg
"""

from __future__ import annotations

import argparse
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import ExperimentConfig
from transforms import get_inference_transform
from model import get_model
from utils import get_device, load_checkpoint


def _to_numpy_rgb(image) -> np.ndarray:
    """Normalizes any of {PIL.Image, numpy array, raw bytes} to an HWC RGB
    uint8 numpy array, which is what Albumentations transforms expect."""
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image)).convert("RGB")
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    if isinstance(image, np.ndarray):
        if image.ndim == 2:  # grayscale -> RGB
            image = np.stack([image] * 3, axis=-1)
        if image.shape[-1] == 4:  # RGBA -> RGB
            image = image[..., :3]
        return image
    raise TypeError(f"Unsupported image type: {type(image)}")


class Predictor:
    """Loads once at API startup; `predict()` is called per-request."""

    def __init__(self, cfg: ExperimentConfig, checkpoint_path: str):
        self.cfg = cfg
        self.device = get_device(cfg.train.device)
        self.model = get_model(
            cfg.model.architecture, num_classes=cfg.data.num_classes,
            pretrained=False, dropout=cfg.model.dropout,
        ).to(self.device)

        checkpoint = load_checkpoint(checkpoint_path, self.model, map_location=self.device)
        self.class_names = checkpoint.get("extra", {}).get("class_names", cfg.data.class_names)
        self.model.eval()

        self.transform = get_inference_transform(cfg.data.image_size, cfg.data.mean, cfg.data.std)

    @torch.no_grad()
    def predict(self, image, top_k: int = 5) -> dict:
        """
        image: PIL.Image, numpy array (HWC, RGB or BGR-as-RGB -- caller's
        responsibility to convert BGR->RGB if the source was OpenCV), or
        raw bytes (e.g. an uploaded file's .read()).
        """
        rgb_array = _to_numpy_rgb(image)
        tensor = self.transform(image=rgb_array)["image"].unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        top_k = min(top_k, len(probs))
        top_indices = np.argsort(probs)[::-1][:top_k]

        return {
            "predicted_class": self.class_names[top_indices[0]],
            "confidence": float(probs[top_indices[0]]),
            "top_k": [
                {"class": self.class_names[i], "confidence": float(probs[i])} for i in top_indices
            ],
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    cfg = ExperimentConfig.load(args.config)
    predictor = Predictor(cfg, args.checkpoint)
    result = predictor.predict(Image.open(args.image))

    print(f"Prediction: {result['predicted_class']}  (confidence: {result['confidence']:.2%})")
    print("Top-k:")
    for entry in result["top_k"]:
        print(f"  {entry['class']:>8}: {entry['confidence']:.2%}")
