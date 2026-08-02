import torch
import torchvision.models as models
import os


MODEL_PATH = "checkpoints/efficientnet_v2_s_baseline/best.pt"
OUTPUT_PATH = "checkpoints/efficientnet_v2_s_baseline/efficientnet_v2_s.onnx"


device = "cpu"


# Load architecture
model = models.efficientnet_v2_s(
    weights=None
)


# Load your trained weights
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


# Handle different checkpoint formats
if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)


model.eval()
model.to(device)


# Dummy input
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
        "image": {
            0: "batch"
        },
        "prediction": {
            0: "batch"
        }
    }
)


print(
    "Saved:",
    OUTPUT_PATH
)
