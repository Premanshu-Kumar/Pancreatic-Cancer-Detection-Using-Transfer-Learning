"""
VGG16 Transfer Learning Model
===============================
Fine-tunes VGG16 (pre-trained on ImageNet) for binary
cancer detection on medical images.
"""

from typing import Tuple
from pathlib import Path

from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

from .base_model import BaseCancerModel


class VGG16CancerModel(BaseCancerModel):
    """
    VGG16-based cancer detection model.

    Architecture notes:
        - 16 weight layers (13 conv + 3 FC, but we replace the FC)
        - Input: 224 × 224 × 3
        - ~138M parameters (base), but only ~2M trainable with frozen base
        - Fine-tuning: unfreeze last 4 conv layers (block5)
    """

    @property
    def model_name(self) -> str:
        return "vgg16"

    def _create_base_model(self, input_shape: Tuple[int, int, int]) -> Model:
        """Create VGG16 base model pre-trained on ImageNet."""
        return VGG16(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
        )

    def _get_fine_tune_layer(self) -> int:
        """
        Unfreeze the last 4 layers of VGG16 (block5_conv1 onwards).

        VGG16 layers:
            ... block4_pool → block5_conv1 → block5_conv2 → block5_conv3 → block5_pool
        """
        return config.FINE_TUNE_LAYERS.get("vgg16", -4)
