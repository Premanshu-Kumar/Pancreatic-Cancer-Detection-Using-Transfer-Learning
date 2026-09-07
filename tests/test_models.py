"""
Tests for Transfer Learning Models
====================================
"""

from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.model_factory import ModelFactory
from src.models.base_model import BaseCancerModel


class TestModelFactory:
    """Tests for ModelFactory."""

    def test_list_models(self):
        """Test that all three models are registered."""
        models = ModelFactory.list_models()
        assert "vgg16" in models
        assert "resnet50" in models
        assert "inceptionv3" in models
        assert len(models) == 3

    def test_create_valid_model(self):
        """Test creating a model by name."""
        for name in ["vgg16", "resnet50", "inceptionv3"]:
            model = ModelFactory.create(name)
            assert isinstance(model, BaseCancerModel)
            assert model.model_name == name

    def test_create_invalid_model(self):
        """Test that invalid model name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            ModelFactory.create("nonexistent_model")

    def test_create_all(self):
        """Test creating all models at once."""
        all_models = ModelFactory.create_all()
        assert len(all_models) == 3
        for name, model in all_models.items():
            assert isinstance(model, BaseCancerModel)


class TestModelArchitecture:
    """Tests for model build and architecture."""

    @pytest.mark.parametrize("model_name,expected_shape", [
        ("vgg16", (224, 224, 3)),
        ("resnet50", (224, 224, 3)),
        ("inceptionv3", (299, 299, 3)),
    ])
    def test_input_shape(self, model_name, expected_shape):
        """Test that each model has the correct input shape."""
        model = ModelFactory.create(model_name)
        assert model.input_shape == expected_shape

    @pytest.mark.parametrize("model_name", ["vgg16", "resnet50", "inceptionv3"])
    def test_build_model(self, model_name):
        """Test that models build successfully."""
        model = ModelFactory.create(model_name, build=True)
        assert model.model is not None
        # Output should be (None, 1) for binary classification
        assert model.model.output_shape == (None, 1)

    @pytest.mark.parametrize("model_name", ["vgg16", "resnet50", "inceptionv3"])
    def test_compile_model(self, model_name):
        """Test that models compile successfully."""
        model = ModelFactory.create(model_name, build=True, compile=True)
        assert model.model.optimizer is not None

    @pytest.mark.parametrize("model_name", ["vgg16", "resnet50", "inceptionv3"])
    def test_freeze_unfreeze(self, model_name):
        """Test freezing and unfreezing base model layers."""
        model = ModelFactory.create(model_name, build=True)

        # Freeze
        model.freeze_base()
        assert model.base_model.trainable is False

        # Unfreeze
        model.unfreeze_base()
        assert model.base_model.trainable is True


class TestModelPrediction:
    """Tests for model prediction (with random weights)."""

    @pytest.fixture
    def sample_image(self, tmp_path):
        """Create a sample test image."""
        from PIL import Image
        img = Image.fromarray(
            np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        )
        path = tmp_path / "test_scan.png"
        img.save(str(path))
        return str(path)

    @pytest.mark.parametrize("model_name", ["resnet50"])
    def test_prediction_format(self, model_name, sample_image):
        """Test that prediction returns correct format."""
        model = ModelFactory.create(model_name, build=True, compile=True)
        result = model.predict(sample_image)

        assert isinstance(result, dict)
        assert "class" in result
        assert "confidence" in result
        assert "probability" in result
        assert "label_index" in result

        assert result["class"] in ["Normal", "Cancerous"]
        assert 0 <= result["confidence"] <= 100
        assert 0 <= result["probability"] <= 1
        assert result["label_index"] in [0, 1]
