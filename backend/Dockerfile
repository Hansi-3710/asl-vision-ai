FROM python:3.11-slim

# System deps needed by opencv-python / mediapipe at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/uploads ml_pipeline/checkpoints

EXPOSE 8000

# NOTE: this image ships WITHOUT a trained model checkpoint (that's a
# multi-hundred-MB artifact you train separately -- see ml_pipeline/README
# and the root README's Training section). Mount or COPY a real checkpoint
# into ml_pipeline/checkpoints/ before running in production, or the API
# will start but /predict will return 503 until one is present.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
