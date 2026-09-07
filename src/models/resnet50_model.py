"""
ResNet50 Transfer Learning Model
==================================
Fine-tunes ResNet50 (pre-trained on ImageNet) for binary
cancer detection on medical images.
"""

from typing import Tuple
from pathlib import Path

from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

from .base_model import BaseCancerModel


class ResNet50CancerModel(BaseCancerModel):
    """
    ResNet50-based cancer detection model.

    Architecture notes:
        - 50 layers with residual (skip) connections
        - Input: 224 × 224 × 3
        - ~25.6M parameters (base)
        - Residual connections solve vanishing gradient problem
        - Fine-tuning: unfreeze from conv5_block1 (layer 143) onwards
    """

    @property
    def model_name(self) -> str:
        return "resnet50"

    def _create_base_model(self, input_shape: Tuple[int, int, int]) -> Model:
        """Create ResNet50 base model pre-trained on ImageNet."""
        return ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
        )

    def _get_fine_tune_layer(self) -> int:
        """
        Unfreeze from conv5_block1 onwards (layer 143 of 175).

        ResNet50 stages:
            conv1 → conv2_block[1-3] → conv3_block[1-4] →
            conv4_block[1-6] → conv5_block[1-3]
        """
        return config.FINE_TUNE_LAYERS.get("resnet50", 143)
