"""
Tests for FastAPI Web Application
====================================
"""

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def sample_upload_image(tmp_path):
    """Create a sample image for upload testing."""
    img = Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )
    path = tmp_path / "test_upload.png"
    img.save(str(path))
    return str(path)


class TestAppConfig:
    """Tests for app configuration."""

    def test_allowed_extensions(self):
        """Test that allowed extensions are properly configured."""
        import config
        assert ".jpg" in config.ALLOWED_EXTENSIONS
        assert ".jpeg" in config.ALLOWED_EXTENSIONS
        assert ".png" in config.ALLOWED_EXTENSIONS
        assert ".tiff" in config.ALLOWED_EXTENSIONS

    def test_class_names(self):
        """Test class name configuration."""
        import config
        assert config.CLASS_NAMES == ["Normal", "Cancerous"]
        assert config.NUM_CLASSES == 2

    def test_image_configs(self):
        """Test that image configs exist for all models."""
        import config
        assert "vgg16" in config.IMAGE_CONFIGS
        assert "resnet50" in config.IMAGE_CONFIGS
        assert "inceptionv3" in config.IMAGE_CONFIGS


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_compute_metrics(self):
        """Test metric computation with known values."""
        from src.evaluation.metrics import compute_metrics

        # Perfect predictions
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.95])

        metrics = compute_metrics(y_true, y_pred_proba)
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0

    def test_compute_metrics_imperfect(self):
        """Test metrics with imperfect predictions."""
        from src.evaluation.metrics import compute_metrics

        y_true = np.array([0, 0, 1, 1, 0])
        y_pred_proba = np.array([0.1, 0.7, 0.8, 0.3, 0.2])

        metrics = compute_metrics(y_true, y_pred_proba)
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1_score"] <= 1
        assert metrics["confusion_matrix"].shape == (2, 2)
        assert metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"] == 5

    def test_classification_report(self):
        """Test classification report generation."""
        from src.evaluation.metrics import generate_classification_report

        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9])

        report = generate_classification_report(y_true, y_pred_proba)
        assert "Normal" in report
        assert "Cancerous" in report
        assert "accuracy" in report
