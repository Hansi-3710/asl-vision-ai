# ASL Vision AI

A real-time American Sign Language alphabet recognition web app -- webcam
and image-upload inference, a live prediction dashboard, and a fine-tuned
EfficientNetV2 vision model behind a FastAPI backend.

![Next.js](https://img.shields.io/badge/Next.js-14-black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **Live webcam recognition** -- continuous, throttled inference against
  your camera feed, with a real bounding box drawn around your detected
  hand (via MediaPipe) and the predicted letter overlaid live.
- **Image upload** -- drag-and-drop a photo, get a prediction, confidence,
  top-5 alternatives, and inference time back in under a second.
- **Dashboard** -- total predictions, average confidence/latency, most
  predicted letters (chart), and a browsable history.
- **REST API** -- `POST /predict`, `POST /predict-webcam`, `GET /health`,
  `GET /metrics`, `GET /history`.
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

## Project Structure

```
asl-vision-ai/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app entrypoint
│   │   ├── core/                # settings, logging
│   │   ├── db/                  # SQLAlchemy models + session
│   │   ├── schemas/              # Pydantic request/response models
│   │   ├── api/                  # /health, /predict*, /metrics, /history
│   │   └── ml/                   # inference service + hand detector
│   ├── ml_pipeline/               # standalone train/eval pipeline
│   │   ├── config.py, dataset.py, model.py, transforms.py
│   │   ├── train.py, evaluate.py, predict.py
│   │   ├── metrics.py, visualization.py, interpretability.py
│   │   └── configs/, checkpoints/, data/
│   ├── tests/                    # pytest (API + hand-detector math)
│   ├── requirements.txt, Dockerfile, .env.example
├── frontend/
│   ├── app/                      # Next.js App Router pages
│   │   ├── page.tsx              # landing page
│   │   ├── camera/page.tsx       # live webcam recognition
│   │   ├── upload/page.tsx       # image upload
│   │   └── dashboard/page.tsx    # stats + history
│   ├── components/               # navbar, hero, camera/*, upload/*, dashboard/*, ui/*
│   ├── hooks/                    # use-webcam-prediction.ts
│   ├── lib/                      # api client, utils
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
# Backend
cd backend
pytest tests/ -v --cov=app

# Frontend
cd frontend
npm test
```

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | API + model load status |
| `/api/predict` | POST | Multipart image upload → prediction |
| `/api/predict-webcam` | POST | Base64 frame → prediction |
| `/api/metrics` | GET | Aggregate dashboard stats |
| `/api/history` | GET | Paginated prediction history (filterable by source/letter) |

Full interactive docs (Swagger UI) at `/docs` once the backend is running.

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

## Known Limitations

- Hand detection and classification are two separate models; a low-quality
  detection (bad lighting, unusual angle) affects the crop fed to the
  classifier even before classification itself runs.
- The live camera polls the backend on a fixed interval (900ms) rather than
  every frame, by design -- see `hooks/use-webcam-prediction.ts` for the
  reasoning.
- The classifier only recognizes static, isolated alphabet letters -- not
  full ASL.

## Future Improvements

- Client-side hand-tracking preview (MediaPipe's JS/WASM build) for
  instant visual feedback even before the backend round-trip completes.
- WebSocket-based streaming instead of polling for lower end-to-end latency.
- A `/predict` response cache keyed on perceptual image hash, to skip
  redundant inference on near-identical consecutive webcam frames.
- Alembic migrations if the schema grows beyond the current single table.

## License

MIT (code). The ASL Alphabet Dataset has its own license terms on Kaggle --
see `docs/model_card.md`.
