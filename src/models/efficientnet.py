"""
EfficientNetV2 Transfer Learning Model
======================================
Fine-tunes EfficientNetV2-B2 for high-efficiency, medical pancreatic
lesion detection with fused-MBConv layers and progressive learning.
"""

from typing import Tuple
from pathlib import Path

from tensorflow.keras import Model
import tensorflow as tf

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

from .base_model import BaseCancerModel


class EfficientNetV2CancerModel(BaseCancerModel):
    """
    EfficientNetV2-B2 based cancer detection model.

    Architecture highlights:
        - Fused-MBConv layers for faster training and parameter efficiency
        - Input: 260 × 260 × 3 (or 224 × 224 × 3 default)
        - Pre-trained on ImageNet-1k / 21k
        - High parameter efficiency (~8.7M parameters)
    """

    @property
    def model_name(self) -> str:
        return "efficientnetv2"

    def _create_base_model(self, input_shape: Tuple[int, int, int]) -> Model:
        """Create EfficientNetV2 base model pre-trained on ImageNet."""
        try:
            return tf.keras.applications.EfficientNetV2B2(
                weights="imagenet",
                include_top=False,
                input_shape=input_shape,
            )
        except AttributeError:
            # Fallback to EfficientNetB2 if V2 is unavailable in older tf builds
            return tf.keras.applications.EfficientNetB2(
                weights="imagenet",
                include_top=False,
                input_shape=input_shape,
            )

    def _get_fine_tune_layer(self) -> int:
        """Unfreeze top stage of EfficientNetV2 for domain adaptation."""
        return config.FINE_TUNE_LAYERS.get("efficientnetv2", -40)
