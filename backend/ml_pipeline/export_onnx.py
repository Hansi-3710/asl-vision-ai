import torch
import torch.nn as nn
import torchvision.models as models
import os


MODEL_PATH = "checkpoints/efficientnet_v2_s_baseline/best.pt"
OUTPUT_PATH = "checkpoints/efficientnet_v2_s_baseline/efficientnet_v2_s.onnx"

device = "cpu"


class ASLClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = models.efficientnet_v2_s(
            weights=None
        )

        self.backbone.classifier[1] = nn.Linear(
            self.backbone.classifier[1].in_features,
            num_classes
        )

    def forward(self, x):
        return self.backbone(x)



NUM_CLASSES = 29   # <-- use your actual number

model = ASLClassifier(NUM_CLASSES)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)


model.eval()
model.to(device)


dummy_input = torch.randn(
    1,
    3,
    384,
    384
)


print("Exporting ONNX...")


torch.onnx.export(
    model,
    dummy_input,
    OUTPUT_PATH,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["image"],
    output_names=["prediction"],
    dynamic_axes={
        "image": {0: "batch"},
        "prediction": {0: "batch"}
    }
)


print("Saved:", OUTPUT_PATH)
