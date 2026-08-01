"""
main.py
=======
FastAPI application entrypoint. Run with:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings, ensure_runtime_dirs
from app.core.logging_config import configure_logging, get_logger
from app.db.database import init_db
from app.ml.inference import InferenceService
from app.api import health, predict, metrics, history

settings = get_settings()
configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    ensure_runtime_dirs(settings)
    init_db()

    inference_service = InferenceService()
    inference_service.load()
    app.state.inference_service = inference_service

    logger.info(f"{settings.APP_NAME} started (environment={settings.ENVIRONMENT})")
    yield
    # --- Shutdown ---
    logger.info(f"{settings.APP_NAME} shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    description="Real-time and upload-based American Sign Language alphabet recognition.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(predict.router, prefix="/api")
app.include_router(metrics.router, prefix="/api")
app.include_router(history.router, prefix="/api")


@app.get("/")
def root():
    return {"name": settings.APP_NAME, "docs": "/docs", "health": "/api/health"}
