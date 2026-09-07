"""
Prediction API Router
======================
Handles image upload, preprocessing, model inference,
and optional Grad-CAM visualization.
"""

import uuid
import base64
import io
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

router = APIRouter(tags=["prediction"])

APP_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = APP_DIR / "uploads"
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


def _get_loaded_models():
    """Import the global model cache from main."""
    from app.main import _loaded_models
    return _loaded_models


@router.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    model_name: str = Form(default="resnet50"),
    generate_gradcam: bool = Form(default=False),
):
    """
    Upload an image and get a cancer classification prediction.

    Parameters
    ----------
    file : UploadFile
        The medical image to classify.
    model_name : str
        Which model to use ('vgg16', 'resnet50', 'inceptionv3').
    generate_gradcam : bool
        Whether to generate a Grad-CAM visualization.

    Returns
    -------
    TemplateResponse
        Results page with prediction details.
    """
    loaded_models = _get_loaded_models()

    # Validate model availability
    if model_name not in loaded_models:
        available = list(loaded_models.keys())
        if not available:
            return templates.TemplateResponse(
                "results.html",
                {
                    "request": request,
                    "error": "No trained models available. Please train a model first.",
                    "prediction": None,
                },
            )
        model_name = available[0]

    # Validate file type
    ext = Path(file.filename).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "error": f"Invalid file type '{ext}'. Allowed: {', '.join(config.ALLOWED_EXTENSIONS)}",
                "prediction": None,
            },
        )

    # Save uploaded file
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{file_id}{ext}"

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    try:
        model = loaded_models[model_name]
        result = model.predict(str(save_path))

        # Encode image to base64 for display
        img = Image.open(save_path).convert("RGB")
        img.thumbnail((400, 400))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Generate Grad-CAM if requested
        gradcam_b64 = None
        if generate_gradcam:
            try:
                from src.utils.gradcam import GradCAM
                cam = GradCAM(model.model, model_name=model_name)
                heatmap, overlay, _ = cam.generate(str(save_path))

                overlay_img = Image.fromarray(overlay)
                overlay_img.thumbnail((400, 400))
                buf = io.BytesIO()
                overlay_img.save(buf, format="PNG")
                gradcam_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception as e:
                print(f"  ⚠ Grad-CAM error: {e}")

        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "prediction": result,
                "model_name": model_name.upper(),
                "image_b64": img_b64,
                "gradcam_b64": gradcam_b64,
                "filename": file.filename,
                "error": None,
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "error": f"Prediction failed: {str(e)}",
                "prediction": None,
            },
        )
    finally:
        # Cleanup uploaded file
        if save_path.exists():
            save_path.unlink()


@router.post("/predict/api")
async def predict_api(
    file: UploadFile = File(...),
    model_name: str = Form(default="resnet50"),
):
    """
    JSON API endpoint for programmatic predictions.

    Returns
    -------
    JSONResponse
        {
            "class": "Normal" or "Cancerous",
            "confidence": 95.23,
            "probability": 0.9523,
            "model": "RESNET50"
        }
    """
    loaded_models = _get_loaded_models()

    if model_name not in loaded_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not available. "
                   f"Available: {list(loaded_models.keys())}",
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {list(config.ALLOWED_EXTENSIONS)}",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{file_id}{ext}"

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    try:
        model = loaded_models[model_name]
        result = model.predict(str(save_path))
        result["model"] = model_name.upper()
        return JSONResponse(content=result)
    finally:
        if save_path.exists():
            save_path.unlink()
