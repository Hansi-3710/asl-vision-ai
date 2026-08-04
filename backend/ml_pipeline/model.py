"""
model.py
========
Wraps torchvision's EfficientNetV2 (S/M/L) with a replaced classifier head
for the 29 ASL classes. Unlike the from-scratch research variant of this
project, pretrained ImageNet weights are used deliberately here -- this is
a production-quality product, not a from-scratch-training research
constraint, so transfer learning is the right call for data efficiency and
accuracy.

Fine-tuning strategy: the backbone starts FROZEN for
`freeze_backbone_epochs` (only the new classifier head trains), then
unfreezes for full fine-tuning at a lower LR (`backbone_lr_multiplier` in
OptimConfig) than the head. This is the standard transfer-learning recipe:
train the new head against fixed features first so early large gradients
from a randomly-initialized head don't scramble the pretrained backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


_ARCH_FACTORY = {
    "efficientnet_v2_s": (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights),
    "efficientnet_v2_m": (models.efficientnet_v2_m, models.EfficientNet_V2_M_Weights),
    "efficientnet_v2_l": (models.efficientnet_v2_l, models.EfficientNet_V2_L_Weights),
}


class ASLEfficientNetV2(nn.Module):
    def __init__(self, architecture: str, num_classes: int, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        if architecture not in _ARCH_FACTORY:
            raise ValueError(f"Unknown architecture '{architecture}'. Choose from {list(_ARCH_FACTORY)}")

        ctor, weights_enum = _ARCH_FACTORY[architecture]
        weights = weights_enum.DEFAULT if pretrained else None
        self.backbone = ctor(weights=weights)

        # torchvision's EfficientNetV2 classifier is Sequential(Dropout, Linear);
        # replace it with a fresh head for our num_classes and desired dropout.
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        self.architecture = architecture

    def forward(self, x):
        return self.backbone(x)

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freezes/unfreezes every backbone parameter EXCEPT the
        classifier head, which always stays trainable."""
        for name, param in self.backbone.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = trainable

    def backbone_parameters(self):
        return [p for n, p in self.backbone.named_parameters() if not n.startswith("classifier") and p.requires_grad]

    def head_parameters(self):
        return [p for n, p in self.backbone.named_parameters() if n.startswith("classifier")]


def get_model(architecture: str, num_classes: int, pretrained: bool = True, dropout: float = 0.3) -> ASLEfficientNetV2:
    return ASLEfficientNetV2(architecture, num_classes=num_classes, pretrained=pretrained, dropout=dropout)


if __name__ == "__main__":
    from utils import count_parameters, model_size_mb

    for arch in ("efficientnet_v2_s", "efficientnet_v2_m"):
        # pretrained=False here to avoid a network download during this smoke test
        m = get_model(arch, num_classes=29, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = m(x)
        assert out.shape == (2, 29), f"{arch} produced wrong output shape: {out.shape}"

        m.set_backbone_trainable(False)
        trainable_before = sum(p.requires_grad for p in m.backbone.parameters())
        m.set_backbone_trainable(True)
        trainable_after = sum(p.requires_grad for p in m.backbone.parameters())
        assert trainable_after > trainable_before, "set_backbone_trainable(True) should unfreeze more params"

        print(
            f"{arch:>20} | params={count_parameters(m):>10,} | size={model_size_mb(m):7.2f} MB | "
            f"output shape OK"
        )
    print("model.py OK -- both architectures produce correct output shape and freeze/unfreeze correctly.")
