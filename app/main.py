"""
FastAPI Web Application — Cancer Detection System
====================================================
Serves a premium dark-themed web interface for uploading
medical images and receiving cancer classification predictions.
"""

import os
import uuid
import shutil
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

import config
from src.models.model_factory import ModelFactory

# =============================================================================
# Global model cache
# =============================================================================
_loaded_models = {}

APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def load_model_if_available(model_name: str):
    """Attempt to load a saved model. Returns None if not found."""
    model_path = config.MODELS_DIR / f"{model_name}_best.keras"
    phase2_path = config.MODELS_DIR / f"{model_name}_phase2_best.keras"

    path_to_load = None
    if model_path.exists():
        path_to_load = str(model_path)
    elif phase2_path.exists():
        path_to_load = str(phase2_path)

    if path_to_load:
        try:
            model = ModelFactory.create(model_name)
            model.load(path_to_load)
            return model
        except Exception as e:
            print(f"  ⚠ Could not load {model_name}: {e}")
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    print("\n🔬 Cancer Detection System — Starting up...")

    for name in ModelFactory.list_models():
        model = load_model_if_available(name)
        if model:
            _loaded_models[name] = model
            print(f"  ✓ Loaded: {name}")
        else:
            print(f"  ○ Not found: {name} (train first)")

    if not _loaded_models:
        print("  ⚠ No trained models found. Train a model first.")
        print(f"    Run: python -m src.training.trainer --model resnet50")

    yield

    # Cleanup uploads
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    print("\n🔬 Cancer Detection System — Shut down.")


# =============================================================================
# FastAPI App
# =============================================================================
app = FastAPI(
    title="Cancer Detection System",
    description="Pancreatic Cancer Detection Using Transfer Learning",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files & templates
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


# =============================================================================
# Routes
# =============================================================================
@app.get("/")
async def landing(request: Request):
    """Render the initial landing page with the fixed video hero section."""
    return templates.TemplateResponse(
        "landing.html",
        {
            "request": request,
        },
    )


@app.get("/detect")
async def detect(request: Request):
    """Render the 2nd page: Cancer Detection workbench (White-Grey theme)."""
    available_models = list(_loaded_models.keys())
    return templates.TemplateResponse(
        "detect.html",
        {
            "request": request,
            "available_models": available_models,
            "has_models": len(available_models) > 0,
        },
    )


from datetime import datetime, timezone
from app.schemas import HealthResponse, ModelListResponse, ModelInfo


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
async def api_health():
    """System health check and loaded model diagnostic probe."""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        has_gpu = len(gpus) > 0
    except Exception:
        has_gpu = False

    return HealthResponse(
        status="healthy",
        version="2.0.0",
        loaded_models=list(_loaded_models.keys()),
        gpu_available=has_gpu,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/api/models", response_model=ModelListResponse, tags=["models"])
async def api_models():
    """Retrieve metadata and status for all registered transfer learning backbones."""
    descriptions = {
        "resnet50": "Residual Network with skip connections for deep CT feature extraction",
        "vgg16": "Classic sequential architecture with small 3x3 receptive fields",
        "inceptionv3": "Multi-scale parallel convolutional kernels for varying lesion diameters",
        "efficientnet": "High-efficiency fused-MBConv backbone with progressive regularization",
        "efficientnetv2": "High-efficiency fused-MBConv backbone with progressive regularization",
        "convnext": "Modern pure-convolutional vision backbone with 7x7 depthwise filters",
    }
    models_meta = []
    for name in ModelFactory.list_models():
        cfg = config.IMAGE_CONFIGS.get(name, {})
        shape = list(cfg.get("input_shape", (224, 224, 3)))
        models_meta.append(ModelInfo(
            name=name,
            display_name=name.upper(),
            input_shape=shape,
            is_loaded=(name in _loaded_models),
            description=descriptions.get(name, "Deep transfer learning medical backbone"),
        ))
    return ModelListResponse(
        default_model=config.DEFAULT_MODEL,
        total_models=len(models_meta),
        models=models_meta,
    )


# Import and include the prediction routers
from app.routers.prediction import router as prediction_router
from app.routers.predict import router as predict_api_router

app.include_router(prediction_router)
app.include_router(predict_api_router)


# =============================================================================
# CLI entry point
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=config.APP_HOST,
        port=config.APP_PORT,
        reload=True,
    )
