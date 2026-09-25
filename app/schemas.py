"""
Pydantic v2 Schemas for FastAPI Endpoints
=========================================
Strict validation for health, models, prediction inputs, and diagnostic outputs.
"""

from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """System health and runtime status schema."""
    status: str = Field(default="healthy", description="API operational health status")
    version: str = Field(default="2.0.0", description="Application semantic version")
    loaded_models: List[str] = Field(default_factory=list, description="Currently loaded model backbones")
    gpu_available: bool = Field(default=False, description="Whether GPU accelerator is detected")
    timestamp: str = Field(description="ISO 8601 timestamp of health probe")


class ModelInfo(BaseModel):
    """Metadata schema for available vision backbones."""
    name: str = Field(description="Architecture identifier")
    display_name: str = Field(description="Human readable clinical name")
    input_shape: List[int] = Field(description="Expected input tensor dimensions [H, W, C]")
    is_loaded: bool = Field(description="Whether weights are cached in memory")
    description: str = Field(description="Model clinical purpose and architecture notes")


class ModelListResponse(BaseModel):
    """Available models collection response."""
    default_model: str = Field(description="Default backbone used for classification")
    total_models: int = Field(description="Total registered model architectures")
    models: List[ModelInfo] = Field(description="List of available architectures")


class BoundingBox(BaseModel):
    """Lesion bounding box coordinates [x, y, width, height]."""
    x: int
    y: int
    width: int
    height: int


class PredictionResponse(BaseModel):
    """Clinical prediction response schema."""
    prediction_id: str = Field(description="Unique UUID for this inference request")
    class_name: str = Field(description="Diagnostic classification ('Cancerous' or 'Normal')")
    is_cancerous: bool = Field(description="Binary flag for clinical cancer detection")
    probability: float = Field(ge=0.0, le=1.0, description="Raw model malignancy probability")
    confidence: float = Field(ge=0.0, le=1.0, description="Diagnostic classification confidence")
    model_used: str = Field(description="Model backbone employed for inference")
    inference_time_ms: float = Field(description="Inference execution duration in milliseconds")
    optimal_threshold: float = Field(default=0.50, description="Youden J optimal diagnostic threshold")
    bounding_boxes: List[BoundingBox] = Field(default_factory=list, description="Detected lesion bounding boxes")
    heatmap_base64: Optional[str] = Field(default=None, description="Base64 encoded Grad-CAM++ overlay image")


class BatchPredictionItem(BaseModel):
    """Individual slice prediction in multi-slice CT series."""
    slice_index: int
    filename: str
    class_name: str
    is_cancerous: bool
    probability: float
    confidence: float
    anomaly_rank: int


class BatchPredictionResponse(BaseModel):
    """Aggregated multi-slice CT batch analysis response."""
    batch_id: str
    total_slices: int
    cancerous_count: int
    normal_count: int
    highest_probability: float
    top_suspicious_slice: Optional[str]
    slices: List[BatchPredictionItem]
