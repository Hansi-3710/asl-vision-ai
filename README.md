# ASL Vision AI

A real-time American Sign Language recognition web app with **two
recognition modes**: a fine-tuned EfficientNetV2 classifier for static
alphabet letters (webcam and image-upload inference, a live prediction
dashboard), and a **continuous, sentence-level recognizer** (MediaPipe
Holistic landmarks → Transformer encoder → CTC decoder → streamed over a
WebSocket) that translates natural, continuous signing into English
sentences in real time -- no capture button, no isolated letters.

![Next.js](https://img.shields.io/badge/Next.js-14-black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **Live webcam recognition** -- continuous, throttled inference against
  your camera feed, with a real bounding box drawn around your detected
  hand (via MediaPipe) and the predicted letter overlaid live.
- **Live Translate (continuous recognition)** -- sign naturally and see
  full sentences appear as you sign, streamed over a WebSocket from a
  Transformer+CTC model running on MediaPipe Holistic landmarks (pose +
  both hands). **Ships with a model trained on synthetic placeholder
  data only** -- see [Continuous Recognition](#continuous-recognition)
  below before expecting it to recognize real ASL.
- **Image upload** -- drag-and-drop a photo, get a prediction, confidence,
  top-5 alternatives, and inference time back in under a second.
- **Dashboard** -- total predictions, average confidence/latency, most
  predicted letters (chart), and a browsable history.
- **REST + WebSocket API** -- `POST /predict`, `POST /predict-webcam`,
  `WS /ws/stream`, `GET /health`, `GET /metrics`, `GET /history`.
- **Dockerized** -- one-command startup via `docker compose up --build`.

## Architecture

```
                     ┌─────────────────────┐
  Browser  ────────► │  Next.js Frontend   │   (React, Tailwind, Framer Motion)
                     └──────────┬──────────┘
                                │ REST (JSON / multipart)
                     ┌──────────▼──────────┐
                     │   FastAPI Backend   │
                     │  ┌────────────────┐ │
                     │  │ Hand Detector  │ │  MediaPipe HandLandmarker
                     │  │ (bounding box) │ │  -- crops to the hand region
                     │  └───────┬────────┘ │
                     │  ┌───────▼────────┐ │
                     │  │  Classifier    │ │  EfficientNetV2 (PyTorch)
                     │  └────────────────┘ │
                     │  ┌────────────────┐ │
                     │  │  SQLite (via   │ │  every prediction persisted
                     │  │  SQLAlchemy)   │ │
                     │  └────────────────┘ │
                     └─────────────────────┘
```

Every prediction flows through the same `InferenceService`: detect a hand
(if the detector model is available) → crop to it → classify → persist →
return the result, including the bounding box, to the frontend.

**Continuous recognition** is a separate, parallel pipeline -- it does not
touch the alphabet classifier above at all:

```
                     ┌─────────────────────┐
  Browser  ────────► │  Next.js Frontend   │
                     │  MediaPipe Holistic │  runs in-browser (WASM) --
                     │  (pose + 2 hands)   │  only landmark PACKETS leave
                     └──────────┬──────────┘  the browser, never video frames
                                │ WebSocket (JSON landmark packets, ~15fps)
                     ┌──────────▼──────────┐
                     │   FastAPI Backend   │
                     │  ┌────────────────┐ │
                     │  │ StreamingSession│ │  per-connection landmark buffer
                     │  └───────┬────────┘ │
                     │  ┌───────▼────────┐ │
                     │  │ Transformer+CTC │ │  ONNX Runtime (CPU, no GPU)
                     │  │ (ONNX Runtime) │ │
                     │  └───────┬────────┘ │
                     │  ┌───────▼────────┐ │
                     │  │ Rule-based     │ │  glosses -> readable sentence
                     │  │ postprocess    │ │
                     │  └────────────────┘ │
                     └─────────────────────┘
```

See [Continuous Recognition](#continuous-recognition) for the full
picture, including the (important) caveat about what the bundled model
has and hasn't actually been trained on.

## Project Structure

```
asl-vision-ai/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app entrypoint
│   │   ├── core/                # settings, logging
│   │   ├── db/                  # SQLAlchemy models + session
│   │   ├── schemas/              # Pydantic request/response models
│   │   ├── api/                  # /health, /predict*, /metrics, /history, /ws/stream
│   │   └── ml/                   # inference service, hand detector, sequence recognizer
│   ├── ml_pipeline/               # alphabet classifier train/eval pipeline
│   │   ├── config.py, dataset.py, model.py, transforms.py
│   │   ├── train.py, evaluate.py, predict.py
│   │   ├── metrics.py, visualization.py, interpretability.py
│   │   └── configs/, checkpoints/, data/
│   ├── continuous_pipeline/       # continuous (sentence-level) recognition pipeline
│   │   ├── landmark_spec.py       # THE feature-vector layout (source of truth)
│   │   ├── vocab.py, config.py, model.py, dataset.py
│   │   ├── landmark_extraction.py # real video -> cached landmarks (needs WLASL/How2Sign)
│   │   ├── generate_synthetic_dataset.py  # placeholder data for pipeline smoke-testing
│   │   ├── train.py, evaluate.py, export_onnx.py
│   │   └── configs/, checkpoints/, data/
│   ├── tests/                    # pytest (API + hand-detector math + streaming + landmarks)
│   ├── requirements.txt, requirements-training.txt, Dockerfile, .env.example
├── frontend/
│   ├── app/                      # Next.js App Router pages
│   │   ├── page.tsx              # landing page
│   │   ├── camera/page.tsx       # live webcam alphabet recognition
│   │   ├── camera/continuous/page.tsx  # Live Translate (continuous recognition)
│   │   ├── upload/page.tsx       # image upload
│   │   └── dashboard/page.tsx    # stats + history
│   ├── components/               # navbar, hero, camera/*, upload/*, dashboard/*, ui/*
│   ├── hooks/                    # use-webcam-prediction, use-holistic-landmarks, use-continuous-recognition
│   ├── lib/                      # api client, holistic-features (landmark layout, frontend twin), utils
│   ├── types/                    # shared TS types (mirrors backend schemas)
│   ├── __tests__/                # Jest + React Testing Library
│   ├── package.json, Dockerfile, .env.local.example
├── docker-compose.yml
└── docs/model_card.md
```

## Installation

### Backend

```bash
cd backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

**Train a model** (required before `/predict` returns real results --
see `ml_pipeline`'s own docstrings for the full pipeline):

```bash
cd ml_pipeline
# download the ASL Alphabet Dataset from Kaggle into data/asl_alphabet_train/
python -c "from config import ExperimentConfig; ExperimentConfig().save('configs/baseline.yaml')"
python train.py --config configs/baseline.yaml
# best checkpoint lands at checkpoints/efficientnet_v2_s_baseline/best.pt,
# which is exactly where app/core/config.py's default ML_CHECKPOINT_PATH expects it.
```

**Enable hand detection** (optional but recommended -- powers the live
bounding box):

```bash
mkdir -p ml_pipeline/checkpoints/mediapipe
curl -L -o ml_pipeline/checkpoints/mediapipe/hand_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

**Run the API:**

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API starts even without a trained checkpoint or hand-detector model --
it just returns clear errors (`503` for no classifier, full-frame fallback
for no hand detector) until you add them.

**Continuous recognition (`/ws/stream`) needs no extra setup** -- a
synthetic-data smoke-test checkpoint ships in `continuous_pipeline/checkpoints/`
already exported to ONNX, so the WebSocket is live as soon as the backend
starts. It just won't recognize real ASL until you train it on real data
-- see [Continuous Recognition](#continuous-recognition) for that.

### Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Visit `http://localhost:3000`.

### Docker (both services at once)

```bash
docker compose up --build
```

Frontend: `http://localhost:3000`. Backend docs: `http://localhost:8000/docs`.

## Testing

```bash
# Backend (includes the alphabet classifier API, hand-detector math,
# streaming WebSocket protocol, landmark parsing, and CTC decode logic)
cd backend
pytest tests/ -v --cov=app

# Frontend
cd frontend
npm test
```

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | API + model load status (both alphabet classifier and continuous model) |
| `/api/predict` | POST | Multipart image upload → prediction |
| `/api/predict-webcam` | POST | Base64 frame → prediction |
| `/ws/stream` | WebSocket | Continuous recognition: stream landmark packets in, get live transcript updates back |
| `/api/metrics` | GET | Aggregate dashboard stats |
| `/api/history` | GET | Paginated prediction history (filterable by source/letter) |

Full interactive docs (Swagger UI) at `/docs` once the backend is running.

## Continuous Recognition

The alphabet classifier above recognizes one static letter per image. This
is a different, parallel system: it recognizes **continuous, natural
signing** -- full sentences, streamed in real time -- built around
MediaPipe Holistic landmarks (pose + both hands, 258 features/frame) fed
into a Transformer encoder with a CTC decoder head. It's what powers the
"Live Translate" page in the frontend (`/camera/continuous`).

### The one caveat that matters most

**The bundled `continuous_pipeline/checkpoints/landmark_transformer_synthetic/`
model was trained on synthetic, randomly-generated placeholder data --
not real ASL.** Training a real continuous-ASL model needs a dataset like
[WLASL](https://dxli94.github.io/WLASL/) or [How2Sign](https://how2sign.github.io/):
tens of thousands of labeled sign-language video clips, each requiring you
to accept dataset-specific license terms, which can't be automated. What's
shipped here proves the entire pipeline runs correctly end to end --
landmark extraction, training, ONNX export, WebSocket serving, the
frontend UI -- with a model that will confidently produce nonsense on a
real webcam, because it never saw real sign language. `is_synthetic_placeholder: true`
is surfaced through `/api/health`, the WebSocket's `ready` message, and a
persistent banner in the frontend precisely so this is never silently
hidden from you.

### How to train it on real data

```bash
cd backend/continuous_pipeline

# 1. Get WLASL or How2Sign yourself (license acceptance required) and note
#    where the video files + a {filename: [gloss, ...]} labels JSON live.

# 2. Download the MediaPipe Holistic model asset (same pattern as the
#    alphabet classifier's hand_landmarker.task):
mkdir -p checkpoints/mediapipe
curl -L -o checkpoints/mediapipe/holistic_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task

# 3. Extract landmarks from the real videos into the cache format train.py expects:
python landmark_extraction.py \
    --videos-dir /path/to/wlasl/videos \
    --labels /path/to/wlasl_labels.json \
    --out data/wlasl_cache

# 4. Train (configs/wlasl_template.yaml is sized for WLASL's ~2000 glosses;
#    copy and point it at data/wlasl_cache from step 3, adjust as needed):
python train.py --config configs/wlasl_template.yaml

# 5. Evaluate (Word Error Rate) and export for serving:
python evaluate.py --config configs/wlasl_template.yaml \
    --checkpoint checkpoints/landmark_transformer_wlasl/best.pt
python export_onnx.py --config configs/wlasl_template.yaml \
    --checkpoint checkpoints/landmark_transformer_wlasl/best.pt \
    --output checkpoints/landmark_transformer_wlasl/model.onnx

# 6. Point the backend at it (backend/.env or app/core/config.py):
#    SEQUENCE_MODEL_ONNX_PATH=continuous_pipeline/checkpoints/landmark_transformer_wlasl/model.onnx
#    SEQUENCE_MODEL_VOCAB_PATH=continuous_pipeline/checkpoints/landmark_transformer_wlasl/vocab.json
#    SEQUENCE_MODEL_IS_PLACEHOLDER=false
```

To re-run the synthetic smoke test yourself (proves the pipeline still
works after any change, in under a minute, no real data needed):

```bash
cd backend/continuous_pipeline
python generate_synthetic_dataset.py --out data/synthetic --num-clips 300
python train.py --config configs/synthetic.yaml
python evaluate.py --config configs/synthetic.yaml --checkpoint checkpoints/landmark_transformer_synthetic/best.pt
python export_onnx.py --config configs/synthetic.yaml \
    --checkpoint checkpoints/landmark_transformer_synthetic/best.pt \
    --output checkpoints/landmark_transformer_synthetic/model.onnx
```

### Known simplifications (deliberate, not oversights)

- **Greedy CTC decode, not beam search.** Simpler, and the doc's "top-3
  alternatives" idea would need beam search to be meaningful -- greedy
  decode only ever has one hypothesis. `continuous_pipeline/model.py` and
  `app/ml/sequence_recognizer.py`'s docstrings note this as a reasonable
  future extension.
- **Whole-buffer re-decode, not true incremental streaming.** Each
  inference tick re-decodes everything currently in the session's buffer
  rather than maintaining CTC state across chunks. Simpler and
  correct-by-construction; costs more compute as a conversation goes on.
  See `StreamingSession`'s docstring in `app/ml/sequence_recognizer.py`.
- **Rule-based sentence post-processing, not a language model.** Turns
  `["hello", "my", "name"]` into `"Hello my name."` (capitalization +
  punctuation only) -- it does NOT insert missing grammar words the way
  the source design doc's "Grammar Correction" example implies (`"hello
  my name john"` → `"Hello, my name is John."` needs real language
  modeling). `postprocess_transcript()` in `app/ml/sequence_recognizer.py`
  is a single, isolated function specifically so it's easy to swap for an
  LLM call later without touching anything else.
- **No facial-grammar landmarks.** Only pose + both hands (258 features/frame)
  are used; MediaPipe Holistic's 468 face landmarks are excluded by
  default (`FACE_ENABLED = False` in `landmark_spec.py`) to keep the
  feature vector a manageable size. Real ASL grammar (yes/no questions,
  negation, topic marking) leans on facial expression more than this
  MVP captures.

## Deployment Notes

- The frontend Dockerfile uses Next.js `output: "standalone"` for a lean
  production image; the backend Dockerfile installs system libs needed by
  OpenCV/MediaPipe at runtime.
- `docker-compose.yml` bind-mounts `ml_pipeline/checkpoints/` so you can
  train on a GPU machine outside Docker and just restart the backend
  container to pick up a new checkpoint -- no rebuild needed.
- `NEXT_PUBLIC_API_URL` is baked in at frontend *build* time (it runs in
  the browser, not the container network) -- update it and rebuild if your
  backend's public URL changes.

## Assumptions Made

Documented explicitly here rather than left implicit in the code:

- **Python 3.11** and **Node.js 18+** are the target runtimes (pinned in
  `requirements.txt` / `package.json` / Dockerfiles).
- **No trained model ships in this repo.** Model weights are a multi-hundred-MB
  training artifact, not source code -- you train one yourself against the
  real ASL Alphabet Dataset (see Installation). The API and frontend both
  handle "no model yet" gracefully (503 / a clear UI message) rather than
  crashing, so the app runs immediately after cloning even before training.
- **MediaPipe's hand-landmarker model file is downloaded separately**, not
  vendored in the repo, for the same reason (binary model asset, not code) --
  one `curl` command in Installation. The app runs without it too; it just
  skips the bounding box and classifies the full frame.
- **SQLite** is used for local development/demo purposes, per the original
  spec. Swapping `DATABASE_URL` to Postgres/MySQL requires no code changes
  elsewhere (see `backend/app/db/database.py`), but SQLite is not intended
  for concurrent multi-user production load.
- **Single-tenant, no authentication.** There's no user account system --
  every prediction is globally visible in `/history` and `/metrics`. Add
  auth before deploying this publicly if that matters for your use case.
- **CORS defaults to `http://localhost:3000`** for local development;
  update `CORS_ORIGINS` in the backend's `.env` for any other deployment.
- **Static alphabet only.** The model recognizes isolated ASL fingerspelling
  letters, not full ASL (grammar, motion, non-manual markers are out of scope).
- **EfficientNetV2-S** is the default architecture (configurable to M/L in
  `ml_pipeline/configs/baseline.yaml`) -- S was chosen as the best
  accuracy/speed default for a real-time webcam demo.
- **The bundled continuous-recognition model is trained on synthetic
  placeholder data, not real ASL** -- see [Continuous Recognition](#continuous-recognition).
  This is the single most important thing to know before evaluating that
  feature; everything else about it (WebSocket protocol, streaming buffer,
  ONNX serving, frontend UI) is real and works, just not sign recognition
  itself yet.
- **Continuous recognition excludes facial landmarks** (pose + both hands
  only, 258 features/frame) -- a deliberate scope cut, not an oversight;
  see `continuous_pipeline/landmark_spec.py`'s docstring.
- **Continuous recognition uses greedy CTC decoding**, not beam search --
  simpler, and there's no "top-3 alternatives" without beam search anyway.

## License

MIT (code). The ASL Alphabet Dataset has its own license terms on Kaggle --
see `docs/model_card.md`.


## Results

*(Generated automatically on 2026-08-01 21:29 UTC by running
`Train_and_Evaluate.ipynb` on Google Colab -- QUICK_MODE (5 epochs).
Every number below was read directly from
`backend/ml_pipeline/outputs/efficientnet_v2_s_baseline_test_results.json` and
`baseline_results.json`, not hand-entered.)*

| Metric | Value |
|---|---|
| Test Accuracy | 0.8142 |
| Macro-F1 | 0.8118 |
| Weighted-F1 | 0.8118 |
| Top-3 Accuracy | 0.9455 |
| Top-5 Accuracy | 0.9734 |
| Calibration (ECE) | 0.1858 |
| Macro ROC-AUC | 0.9916 |
| Non-deep baseline (HOG+SVM) accuracy | 0.7475 |
| Parameters | 20,214,637 |
| Model size | 77.71 MB |
| Inference latency | 16.65 ms/image (cuda) |

**Note:** this was a QUICK_MODE (5-epoch) run for pipeline verification -- accuracy will improve substantially with the full 30-epoch schedule (set QUICK_MODE = False in Step 6 and re-run).
