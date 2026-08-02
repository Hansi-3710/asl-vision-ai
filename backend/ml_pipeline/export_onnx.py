import torch
import torchvision.models as models
import torch.nn as nn
import os


MODEL_PATH = "checkpoints/efficientnet_v2_s_baseline/best.pt"
OUTPUT_PATH = "checkpoints/efficientnet_v2_s_baseline/efficientnet_v2_s.onnx"

device = "cpu"


model = models.efficientnet_v2_s(
    weights=None
)


# 29 ASL classes
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
clean_state_dict = {}

for key, value in state_dict.items():
    if key.startswith("backbone."):
        clean_state_dict[key.replace("backbone.", "")] = value
    else:
        clean_state_dict[key] = value


result = model.load_state_dict(
    clean_state_dict,
    strict=False
)


print("Missing keys:", result.missing_keys)
print("Unexpected keys:", result.unexpected_keys)


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
        "image": {0: "batch"},
        "prediction": {0: "batch"}
    }
)


print("Saved:", OUTPUT_PATH)