"""
interpretability.py
===================
Grad-CAM (Selvaraju et al., 2017) for the ASLEfficientNetV2 wrapper.
Hooks the last convolutional layer inside the backbone's feature extractor
(`model.backbone.features` -- torchvision's EfficientNet exposes its conv
stack under `.features`, same convention as their other CNN model classes)
and uses gradient-weighted feature maps to visualize where the network
looked when making a given prediction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> tuple:
        self.model.eval()
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        confidence = probs[0, target_class].item()

        self.model.zero_grad()
        logits[0, target_class].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, target_class, confidence


def get_last_conv_layer(model: torch.nn.Module) -> torch.nn.Module:
    """
    Finds the last Conv2d in the EfficientNetV2 backbone's feature
    extractor. `model` here is an `ASLEfficientNetV2` instance
    (see model.py) -- its conv stack lives at `model.backbone.features`,
    NOT `model.features` (unlike the from-scratch research CNNs).
    """
    feature_extractor = model.backbone.features
    conv_layers = [m for m in feature_extractor.modules() if isinstance(m, torch.nn.Conv2d)]
    if not conv_layers:
        raise ValueError("No Conv2d layers found in model.backbone.features")
    return conv_layers[-1]


def overlay_heatmap_on_image(original_image: Image.Image, heatmap: np.ndarray, save_path: str, alpha: float = 0.45):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original_image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(original_image)
    axes[2].imshow(heatmap, cmap="jet", alpha=alpha)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
