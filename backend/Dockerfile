FROM python:3.11-slim

# System deps needed by opencv-python / mediapipe at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 user

WORKDIR /srv

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create required folders and give ownership to the non-root user
RUN mkdir -p data/uploads ml_pipeline/checkpoints \
    && chown -R user:user /srv

# Switch to non-root user
USER user

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
