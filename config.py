"""
Centralized Configuration for Cancer Detection System
======================================================
All hyperparameters, paths, and settings are defined here.
Modify this file to customize the training pipeline.
"""

import os
from pathlib import Path

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =============================================================================
# Project Paths
# =============================================================================
BASE_DIR = Path(os.getenv("BASE_DIR", str(Path(__file__).resolve().parent)))
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", str(DATA_DIR / "raw")))
PROCESSED_DATA_DIR = Path(os.getenv("PROCESSED_DATA_DIR", str(DATA_DIR / "processed")))
SAMPLE_DATA_DIR = Path(os.getenv("SAMPLE_DATA_DIR", str(DATA_DIR / "sample")))
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(BASE_DIR / "models")))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", str(BASE_DIR / "results")))
NOTEBOOKS_DIR = Path(os.getenv("NOTEBOOKS_DIR", str(BASE_DIR / "notebooks")))

# Training / Validation / Test subdirectories
TRAIN_DIR = PROCESSED_DATA_DIR / "train"
VAL_DIR = PROCESSED_DATA_DIR / "val"
TEST_DIR = PROCESSED_DATA_DIR / "test"

# =============================================================================
# Class Configuration
# =============================================================================
CLASS_NAMES = ["Normal", "Cancerous"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_MODE = "binary"  # 'binary' for 2-class; 'categorical' for multi-class

# =============================================================================
# Image Configuration (per model)
# =============================================================================
IMAGE_CONFIGS = {
    "vgg16": {
        "img_size": (224, 224),
        "input_shape": (224, 224, 3),
        "preprocess_fn": "tensorflow.keras.applications.vgg16.preprocess_input",
    },
    "resnet50": {
        "img_size": (224, 224),
        "input_shape": (224, 224, 3),
        "preprocess_fn": "tensorflow.keras.applications.resnet50.preprocess_input",
    },
    "inceptionv3": {
        "img_size": (299, 299),
        "input_shape": (299, 299, 3),
        "preprocess_fn": "tensorflow.keras.applications.inception_v3.preprocess_input",
    },
}

# Default image configuration
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "resnet50")
DEFAULT_IMG_SIZE = IMAGE_CONFIGS.get(DEFAULT_MODEL, IMAGE_CONFIGS["resnet50"])["img_size"]
DEFAULT_INPUT_SHAPE = IMAGE_CONFIGS.get(DEFAULT_MODEL, IMAGE_CONFIGS["resnet50"])["input_shape"]

# =============================================================================
# Training Hyperparameters
# =============================================================================
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
EPOCHS_PHASE1 = int(os.getenv("EPOCHS_PHASE1", "15"))          # Feature extraction (frozen base)
EPOCHS_PHASE2 = int(os.getenv("EPOCHS_PHASE2", "20"))          # Fine-tuning (unfrozen top layers)
LEARNING_RATE_PHASE1 = float(os.getenv("LEARNING_RATE_PHASE1", "1e-4"))
LEARNING_RATE_PHASE2 = float(os.getenv("LEARNING_RATE_PHASE2", "1e-5"))

# Optimizer
OPTIMIZER = os.getenv("OPTIMIZER", "adam")

# Loss function
LOSS_FUNCTION = os.getenv("LOSS_FUNCTION", "binary_crossentropy")

# =============================================================================
# Data Split Ratios
# =============================================================================
TRAIN_SPLIT = float(os.getenv("TRAIN_SPLIT", "0.70"))
VAL_SPLIT = float(os.getenv("VAL_SPLIT", "0.15"))
TEST_SPLIT = float(os.getenv("TEST_SPLIT", "0.15"))

# =============================================================================
# Data Augmentation Parameters
# =============================================================================
AUGMENTATION_CONFIG = {
    "rotation_range": int(os.getenv("AUG_ROTATION_RANGE", "20")),
    "width_shift_range": float(os.getenv("AUG_WIDTH_SHIFT", "0.1")),
    "height_shift_range": float(os.getenv("AUG_HEIGHT_SHIFT", "0.1")),
    "shear_range": float(os.getenv("AUG_SHEAR_RANGE", "0.1")),
    "zoom_range": float(os.getenv("AUG_ZOOM_RANGE", "0.2")),
    "horizontal_flip": os.getenv("AUG_HORIZONTAL_FLIP", "true").lower() == "true",
    "vertical_flip": os.getenv("AUG_VERTICAL_FLIP", "false").lower() == "true",
    "brightness_range": [0.8, 1.2],
    "fill_mode": "nearest",
}

# =============================================================================
# Callbacks Configuration
# =============================================================================
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", "7"))
REDUCE_LR_PATIENCE = int(os.getenv("REDUCE_LR_PATIENCE", "3"))
REDUCE_LR_FACTOR = float(os.getenv("REDUCE_LR_FACTOR", "0.2"))
MIN_LEARNING_RATE = float(os.getenv("MIN_LEARNING_RATE", "1e-7"))

# =============================================================================
# Model Fine-Tuning Layers
# =============================================================================
FINE_TUNE_LAYERS = {
    "vgg16": -4,            # Unfreeze last 4 layers
    "resnet50": 143,        # Unfreeze from layer 143 onwards (conv5_block1)
    "inceptionv3": 249,     # Unfreeze from layer 249 onwards (mixed7)
}

# =============================================================================
# Web Application
# =============================================================================
APP_HOST = os.getenv("APP_HOST", os.getenv("HOST", "0.0.0.0"))
APP_PORT = int(os.getenv("APP_PORT", os.getenv("PORT", "8000")))
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# =============================================================================
# Miscellaneous
# =============================================================================
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
VERBOSE = int(os.getenv("VERBOSE", "1"))

# =============================================================================
# Ensure directories exist
# =============================================================================
def setup_directories():
    """Create all required project directories."""
    dirs = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR,
        TRAIN_DIR / "Normal", TRAIN_DIR / "Cancerous",
        VAL_DIR / "Normal", VAL_DIR / "Cancerous",
        TEST_DIR / "Normal", TEST_DIR / "Cancerous",
        MODELS_DIR, RESULTS_DIR, NOTEBOOKS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    setup_directories()
    print("✓ All directories created successfully.")
    print(f"  Base directory: {BASE_DIR}")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Models directory: {MODELS_DIR}")
    print(f"  Results directory: {RESULTS_DIR}")
