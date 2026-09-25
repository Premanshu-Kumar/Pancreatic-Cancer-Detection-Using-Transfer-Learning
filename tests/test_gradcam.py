"""
Unit tests for Grad-CAM and Grad-CAM++ Explainability
====================================================
"""

import numpy as np
import pytest
from src.utils.gradcam import extract_lesion_bounding_boxes


class TestGradCAMUtils:
    """Test suite for Grad-CAM helper and bounding box extraction."""

    def test_extract_bounding_boxes_from_heatmap(self):
        """Verify bounding box extraction identifies high activation lesion regions."""
        heatmap = np.zeros((100, 100), dtype=np.float32)
        # Create a bright lesion region
        heatmap[30:50, 40:70] = 0.85

        boxes = extract_lesion_bounding_boxes(heatmap, threshold=0.5, min_area=20)
        assert len(boxes) >= 1
        x, y, w, h = boxes[0]
        assert 35 <= x <= 45
        assert 25 <= y <= 35
        assert 25 <= w <= 35
        assert 15 <= h <= 25

    def test_extract_bounding_boxes_blank_heatmap(self):
        """Verify no false positive bounding boxes on zero activation."""
        heatmap = np.zeros((100, 100), dtype=np.float32)
        boxes = extract_lesion_bounding_boxes(heatmap, threshold=0.5)
        assert len(boxes) == 0
