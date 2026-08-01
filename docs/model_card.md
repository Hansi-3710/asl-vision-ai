# Model Card: ASL Vision AI Classifier

Following Mitchell et al. (2019), *Model Cards for Model Reporting*. Numeric
fields below are placeholders until you've actually trained a model against
the real dataset -- see the root README's Training section.

## Model Details
- **Architecture:** EfficientNetV2 (S/M/L, configurable), ImageNet-pretrained backbone + fine-tuned classifier head.
- **Framework:** PyTorch / torchvision.
- **Input:** RGB image, resized per config (default 224x224), ImageNet-normalized.
- **Output:** Softmax distribution over 29 classes (A-Z, `space`, `del`, `nothing`).
- **Auxiliary component:** MediaPipe HandLandmarker for hand detection/cropping prior to classification.

## Intended Use
- **Primary use:** A demo/portfolio product recognizing static ASL alphabet
  fingerspelling letters from a webcam feed or an uploaded photo.
- **Out-of-scope:** Not a full ASL translator (ASL includes grammar,
  non-manual markers, and continuous motion far beyond the alphabet) and
  not suitable for safety-critical, medical, legal, or accessibility-critical
  decisions.

## Training Data
ASL Alphabet Dataset (Nagaraj, via Kaggle), ~87,000 images, 29 classes. Split
by signer/session group (not per-image) to avoid near-duplicate-frame
leakage across train/val/test -- see `backend/ml_pipeline/dataset.py`.

## Metrics
`[TO BE FILLED IN AFTER TRAINING -- run backend/ml_pipeline/evaluate.py and
copy accuracy, macro-F1, calibration (ECE), and efficiency numbers here.]`

## Bias, Fairness, and Limitations
- The dataset is captured by a small number of signers in consistent
  studio-like conditions -- accuracy on genuinely different signers,
  hand sizes, skin tones, backgrounds, and lighting is likely lower than
  in-distribution test accuracy suggests.
- Hand detection (MediaPipe) and classification are two separate models;
  detection failures (poor lighting, unusual hand angles) will surface as
  `hand_detected: false` responses, not classification errors -- worth
  distinguishing when debugging low-confidence results.

## Ethical Considerations
This is a research/portfolio demo of a narrow sub-problem (isolated letter
recognition), not a substitute for a qualified ASL interpreter or a
production accessibility tool. Treat any accuracy number as an upper bound
on real-world performance given the training data's limited diversity.
