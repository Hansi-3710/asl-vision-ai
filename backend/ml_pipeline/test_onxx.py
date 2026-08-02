import onnx
import onnxruntime as ort
import numpy as np


MODEL_PATH = "checkpoints/efficientnet_v2_s_baseline/efficientnet_v2_s.onnx"


print("Loading ONNX model...")

# Check model structure
model = onnx.load(MODEL_PATH)
onnx.checker.check_model(model)

print("ONNX structure OK")


# Create inference session
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

print("ONNX Runtime loaded")


# Get input name
input_name = session.get_inputs()[0].name

print("Input:", input_name)


# Random fake image
dummy_image = np.random.randn(
    1,
    3,
    384,
    384
).astype(np.float32)


# Run prediction
output = session.run(
    None,
    {
        input_name: dummy_image
    }
)


prediction = output[0]


print("Output shape:", prediction.shape)
print("Prediction vector:")
print(prediction)

print("TEST PASSED ✅")
