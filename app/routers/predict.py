"""
Asynchronous Prediction API Router with Pydantic v2 Validation
==============================================================
Provides high-performance async inference endpoints for single CT slices,
returning validated diagnostic probabilities and Grad-CAM++ heatmaps.
"""

import time
import uuid
import base64
import io
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config
from app.schemas import PredictionResponse, BoundingBox, BatchPredictionResponse, BatchPredictionItem
from src.data.preprocessing import preprocess_image
from src.utils.gradcam import GradCAMPlusPlus, extract_lesion_bounding_boxes

router = APIRouter(prefix="/api/predict", tags=["prediction"])


def _get_model(model_name: str):
    """Retrieve model instance from global cache or load from disk."""
    from app.main import _loaded_models
    from app.main import load_model_if_available

    name = model_name.lower().strip()
    if name in _loaded_models:
        return _loaded_models[name]

    model = load_model_if_available(name)
    if model:
        _loaded_models[name] = model
        return model

    raise HTTPException(
        status_code=404,
        detail=f"Model '{model_name}' is not loaded and weights were not found. Please train model first."
    )


@router.post("", response_model=PredictionResponse)
async def api_predict_image(
    file: UploadFile = File(..., description="Abdominal CT slice image (PNG/JPEG)"),
    model_name: str = Form(default="resnet50"),
    generate_gradcam: bool = Form(default=True),
    optimal_threshold: float = Form(default=0.50),
):
    """
    Async diagnostic inference on a single CT slice with Pydantic v2 response validation.
    """
    start_time = time.perf_counter()
    pred_id = str(uuid.uuid4())

    # Read image contents
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        raw_arr = np.array(pil_img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # Load model
    model = _get_model(model_name)
    target_size = model.img_size

    # Preprocess
    img_resized = cv2.resize(raw_arr, target_size, interpolation=cv2.INTER_LANCZOS4).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(img_resized, axis=0)

    # Inference
    prob = float(model.predict(input_tensor)[0])
    is_cancer = prob >= optimal_threshold
    conf = prob if is_cancer else (1.0 - prob)
    cls_name = "Cancerous" if is_cancer else "Normal"

    # Optional Grad-CAM++
    heatmap_b64 = None
    bounding_boxes = []
    if generate_gradcam:
        try:
            cam = GradCAMPlusPlus(model.model, model_name=model_name)
            heatmap = cam.compute_heatmap(input_tensor, pred_index=0)
            raw_boxes = extract_lesion_bounding_boxes(heatmap, threshold=0.55)
            bounding_boxes = [BoundingBox(x=b[0], y=b[1], width=b[2], height=b[3]) for b in raw_boxes]

            # Generate overlay
            heatmap_resized = cv2.resize(heatmap, (raw_arr.shape[1], raw_arr.shape[0]))
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            blended = np.uint8(raw_arr * 0.6 + heatmap_color * 0.4)

            # Convert overlay to base64
            buffered = io.BytesIO()
            Image.fromarray(blended).save(buffered, format="JPEG", quality=85)
            heatmap_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception:
            pass

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    return PredictionResponse(
        prediction_id=pred_id,
        class_name=cls_name,
        is_cancerous=is_cancer,
        probability=round(prob, 4),
        confidence=round(conf, 4),
        model_used=model_name,
        inference_time_ms=round(elapsed_ms, 2),
        optimal_threshold=optimal_threshold,
        bounding_boxes=bounding_boxes,
        heatmap_base64=heatmap_b64,
    )


@router.post("/batch", response_model=BatchPredictionResponse)
async def api_predict_batch(
    files: List[UploadFile] = File(..., description="CT sequence slices to batch analyze"),
    model_name: str = Form(default="resnet50"),
    optimal_threshold: float = Form(default=0.50),
):
    """
    Analyze full CT series slices and rank them by anomaly suspicion score.
    """
    batch_id = str(uuid.uuid4())
    model = _get_model(model_name)
    target_size = model.img_size

    slice_results = []
    cancerous_count = 0
    normal_count = 0

    for idx, f in enumerate(files):
        content = await f.read()
        if not content:
            continue
        try:
            pil_img = Image.open(io.BytesIO(content)).convert("RGB")
            raw_arr = np.array(pil_img)
            img_resized = cv2.resize(raw_arr, target_size, interpolation=cv2.INTER_LANCZOS4).astype(np.float32) / 255.0
            input_tensor = np.expand_dims(img_resized, axis=0)
            prob = float(model.predict(input_tensor)[0])
            is_cancer = prob >= optimal_threshold
            conf = prob if is_cancer else (1.0 - prob)

            if is_cancer:
                cancerous_count += 1
            else:
                normal_count += 1

            slice_results.append({
                "slice_index": idx + 1,
                "filename": f.filename or f"slice_{idx+1}.png",
                "class_name": "Cancerous" if is_cancer else "Normal",
                "is_cancerous": is_cancer,
                "probability": round(prob, 4),
                "confidence": round(conf, 4),
            })
        except Exception:
            continue

    # Sort descending by cancer probability for triage ranking
    slice_results.sort(key=lambda s: s["probability"], reverse=True)
    for rank, item in enumerate(slice_results, 1):
        item["anomaly_rank"] = rank

    highest_prob = slice_results[0]["probability"] if slice_results else 0.0
    top_slice = slice_results[0]["filename"] if slice_results else None

    return BatchPredictionResponse(
        batch_id=batch_id,
        total_slices=len(slice_results),
        cancerous_count=cancerous_count,
        normal_count=normal_count,
        highest_probability=round(highest_prob, 4),
        top_suspicious_slice=top_slice,
        slices=[BatchPredictionItem(**item) for item in slice_results],
    )

