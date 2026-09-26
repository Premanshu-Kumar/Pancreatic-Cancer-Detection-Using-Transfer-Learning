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


def get_model(model_name: str):
    """Retrieve model instance from global cache or load from disk on-demand."""
    name = model_name.lower().strip()
    if name in _loaded_models:
        return _loaded_models[name]

    # Free memory if another model is already loaded (keeps RAM under 512MB on Render Free Tier)
    if _loaded_models:
        _loaded_models.clear()
        import gc
        gc.collect()

    model = load_model_if_available(name)
    if model:
        _loaded_models[name] = model
        return model
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Fast, low-memory startup with on-demand model loading."""
    print("\n🔬 Cancer Detection System — Starting up...")

    available_checkpoints = []
    for name in ModelFactory.list_models():
        model_path = config.MODELS_DIR / f"{name}_best.keras"
        phase2_path = config.MODELS_DIR / f"{name}_phase2_best.keras"
        if model_path.exists() or phase2_path.exists():
            available_checkpoints.append(name)
            print(f"  ✓ Model checkpoint available (on-demand): {name}")
        else:
            print(f"  ○ Checkpoint not found: {name}")

    if not available_checkpoints:
        print("  ⚠ No trained model checkpoints found. Train a model first.")
        print("    Run: python -m src.training.trainer --model resnet50")
    else:
        print(f"  ✓ Ready with {len(available_checkpoints)} available model(s).")

    yield

    # Cleanup uploads and model references on shutdown
    if UPLOAD_DIR.exists():
        shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    _loaded_models.clear()
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
        loaded_models=list(_loaded_models.keys()) or [
            name for name in ModelFactory.list_models()
            if (config.MODELS_DIR / f"{name}_best.keras").exists() or (config.MODELS_DIR / f"{name}_phase2_best.keras").exists()
        ],
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
        has_checkpoint = (
            (config.MODELS_DIR / f"{name}_best.keras").exists()
            or (config.MODELS_DIR / f"{name}_phase2_best.keras").exists()
        )
        models_meta.append(ModelInfo(
            name=name,
            display_name=name.upper(),
            input_shape=shape,
            is_loaded=(name in _loaded_models or has_checkpoint),
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
