"""
Tests for Image Preprocessing Pipeline
========================================
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocessing import (
    preprocess_image,
    preprocess_directory,
    validate_image,
    get_image_stats,
)


@pytest.fixture
def sample_image_path(tmp_path):
    """Create a temporary test image."""
    img = Image.fromarray(
        np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
    )
    path = tmp_path / "test_image.png"
    img.save(str(path))
    return str(path)


@pytest.fixture
def sample_image_dir(tmp_path):
    """Create a directory with multiple test images."""
    for i in range(5):
        img = Image.fromarray(
            np.random.randint(0, 255, (80 + i * 10, 80 + i * 10, 3), dtype=np.uint8)
        )
        img.save(str(tmp_path / f"img_{i}.png"))
    return str(tmp_path)


class TestPreprocessImage:
    """Tests for the preprocess_image function."""

    def test_basic_preprocessing(self, sample_image_path):
        """Test basic image loading and preprocessing."""
        result = preprocess_image(sample_image_path, target_size=(224, 224))
        assert isinstance(result, np.ndarray)
        assert result.shape == (224, 224, 3)
        assert result.dtype == np.float32

    def test_normalization(self, sample_image_path):
        """Test that normalized images have values in [0, 1]."""
        result = preprocess_image(
            sample_image_path, target_size=(224, 224), normalize=True
        )
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_no_normalization(self, sample_image_path):
        """Test preprocessing without normalization."""
        result = preprocess_image(
            sample_image_path, target_size=(224, 224), normalize=False
        )
        assert result.max() > 1.0  # Pixel values should be 0-255

    def test_denoising(self, sample_image_path):
        """Test that denoising produces a valid output."""
        result = preprocess_image(
            sample_image_path, target_size=(224, 224), denoise=True
        )
        assert result.shape == (224, 224, 3)

    def test_different_sizes(self, sample_image_path):
        """Test resizing to different dimensions."""
        for size in [(224, 224), (299, 299), (128, 128)]:
            result = preprocess_image(sample_image_path, target_size=size)
            assert result.shape[:2] == size

    def test_pil_output(self, sample_image_path):
        """Test returning a PIL Image instead of array."""
        result = preprocess_image(
            sample_image_path, target_size=(224, 224), as_array=False
        )
        assert isinstance(result, Image.Image)

    def test_invalid_path(self):
        """Test that invalid path raises ValueError."""
        with pytest.raises(ValueError, match="Could not load image"):
            preprocess_image("nonexistent_image.png")


class TestPreprocessDirectory:
    """Tests for the preprocess_directory function."""

    def test_directory_preprocessing(self, sample_image_dir, tmp_path):
        """Test preprocessing all images in a directory."""
        output_dir = tmp_path / "output"
        count = preprocess_directory(
            sample_image_dir, str(output_dir), target_size=(224, 224)
        )
        assert count == 5
        assert len(list(output_dir.iterdir())) == 5

    def test_output_format(self, sample_image_dir, tmp_path):
        """Test output format specification."""
        output_dir = tmp_path / "output_jpg"
        preprocess_directory(
            sample_image_dir, str(output_dir),
            target_size=(224, 224), output_format="jpg"
        )
        for f in output_dir.iterdir():
            assert f.suffix == ".jpg"


class TestValidateImage:
    """Tests for the validate_image function."""

    def test_valid_image(self, sample_image_path):
        """Test validation of a valid image."""
        assert validate_image(sample_image_path) is True

    def test_invalid_image(self, tmp_path):
        """Test validation of a non-image file."""
        bad_file = tmp_path / "not_image.txt"
        bad_file.write_text("This is not an image")
        assert validate_image(str(bad_file)) is False


class TestGetImageStats:
    """Tests for the get_image_stats function."""

    def test_basic_stats(self, sample_image_dir):
        """Test computing image statistics."""
        stats = get_image_stats(sample_image_dir)
        assert stats["count"] == 5
        assert len(stats["sizes"]) == 5
        assert isinstance(stats["mean_size"], tuple)
        assert len(stats["formats"]) > 0
