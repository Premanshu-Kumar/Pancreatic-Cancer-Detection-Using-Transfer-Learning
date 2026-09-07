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

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
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
async def index(request: Request):
    """Render the main upload/prediction page."""
    available_models = list(_loaded_models.keys())
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "available_models": available_models,
            "has_models": len(available_models) > 0,
        },
    )


# Import and include the prediction router
from app.routers.prediction import router as prediction_router
app.include_router(prediction_router)


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
