import torch
import torchvision.models as models
import torch.nn as nn
import os


MODEL_PATH = "checkpoints/efficientnet_v2_s_baseline/best.pt"
OUTPUT_PATH = "checkpoints/efficientnet_v2_s_baseline/efficientnet_v2_s.onnx"


device = "cpu"


# Create same architecture
model = models.efficientnet_v2_s(
    weights=None
)


# Replace classifier for 29 ASL classes
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    29
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


state_dict = checkpoint["model_state_dict"]


# Remove "backbone." prefix
new_state_dict = {}

for key, value in state_dict.items():
    if key.startswith("backbone."):
        new_key = key.replace("backbone.", "")
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value


# Load weights
missing, unexpected = model.load_state_dict(
    new_state_dict,
    strict=False
)


print("Missing keys:", missing)
print("Unexpected keys:", unexpected)


model.eval()
model.to(device)


dummy_input = torch.randn(
    1,
    3,
    384,
    384
)


os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
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


print("Saved:", OUTPUT_PATH)
