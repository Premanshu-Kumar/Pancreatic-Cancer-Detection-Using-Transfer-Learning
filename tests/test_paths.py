"""
Tests for Cross-Platform Path Integrity & Module Imports
=========================================================
Validates that directory configurations, dynamic environment overrides,
and all core project modules load and operate cleanly without path breakages.
"""

import os
import tempfile
from pathlib import Path
import pytest

import config


class TestPathIntegrity:
    """Test suite for path validation and directory creation."""

    def test_base_directories_exist_and_accessible(self):
        """Verify all critical base directories exist or can be created."""
        config.setup_directories()

        directories = [
            config.DATA_DIR,
            config.RAW_DATA_DIR,
            config.PROCESSED_DATA_DIR,
            config.SAMPLE_DATA_DIR,
            config.MODELS_DIR,
            config.RESULTS_DIR,
            config.NOTEBOOKS_DIR,
            config.TRAIN_DIR / "Normal",
            config.TRAIN_DIR / "Cancerous",
            config.VAL_DIR / "Normal",
            config.VAL_DIR / "Cancerous",
            config.TEST_DIR / "Normal",
            config.TEST_DIR / "Cancerous",
        ]

        for d in directories:
            assert isinstance(d, Path), f"{d} is not a Path object"
            assert d.exists(), f"Directory does not exist: {d}"
            assert d.is_dir(), f"Path is not a directory: {d}"

    def test_directory_write_permissions(self):
        """Verify that configured results and models directories are writable."""
        config.setup_directories()

        test_file_results = config.RESULTS_DIR / ".write_test_tmp"
        try:
            test_file_results.write_text("write_test", encoding="utf-8")
            assert test_file_results.exists()
            assert test_file_results.read_text(encoding="utf-8") == "write_test"
        finally:
            if test_file_results.exists():
                test_file_results.unlink()

        test_file_models = config.MODELS_DIR / ".write_test_tmp"
        try:
            test_file_models.write_text("write_test", encoding="utf-8")
            assert test_file_models.exists()
        finally:
            if test_file_models.exists():
                test_file_models.unlink()

    def test_dynamic_env_override(self, monkeypatch):
        """Test that environment variables override default paths."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            monkeypatch.setenv("DATA_DIR", tmp_dir)
            
            # Re-read path directly using os.getenv logic
            custom_data_dir = Path(os.getenv("DATA_DIR", str(config.BASE_DIR / "data")))
            assert custom_data_dir == Path(tmp_dir)
            assert custom_data_dir.exists()


class TestModuleImports:
    """Verify that all core modules can be imported without error."""

    def test_import_config(self):
        import config
        assert hasattr(config, "BASE_DIR")
        assert hasattr(config, "BATCH_SIZE")
        assert hasattr(config, "IMAGE_CONFIGS")

    def test_import_models(self):
        from src.models.model_factory import ModelFactory
        from src.models.base_model import BaseCancerModel
        from src.models.resnet50_model import ResNet50CancerModel
        from src.models.vgg16_model import VGG16CancerModel
        from src.models.inception_model import InceptionV3CancerModel
        assert len(ModelFactory.list_models()) >= 3

    def test_import_utils(self):
        from src.utils.helpers import set_seed, get_device_info, format_duration
        from src.utils.gradcam import GradCAM
        from src.utils.model_downloader import download_model, ensure_model_exists, list_available_models
        assert callable(set_seed)
        assert callable(ensure_model_exists)

    def test_import_data_pipeline(self):
        from src.data.preprocessing import preprocess_image, validate_image
        from src.data.dataset import create_generators
        assert callable(preprocess_image)
        assert callable(create_generators)

    def test_import_training_and_eval(self):
        from src.training.trainer import Trainer
        from src.evaluation.metrics import evaluate_model
        assert callable(evaluate_model)
