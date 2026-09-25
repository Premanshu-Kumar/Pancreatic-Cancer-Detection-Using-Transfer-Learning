"""
ConvNeXt Modern Vision Backbone
===============================
Pure convolutional model re-architected with vision transformer design
principles (7x7 depthwise convolutions, inverted bottlenecks, LayerNorm).
"""

from typing import Tuple
from pathlib import Path

from tensorflow.keras import Model
import tensorflow as tf

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

from .base_model import BaseCancerModel


class ConvNeXtCancerModel(BaseCancerModel):
    """
    ConvNeXtBase / ConvNeXtTiny cancer detection model.

    Architecture highlights:
        - 7x7 depthwise convolutions for expanded receptive field
        - Inverted bottleneck structure similar to Swin Transformer
        - LayerNorm instead of BatchNorm for stabilized medical feature maps
        - Exceptional fine-grained feature representation for tumor boundaries
    """

    @property
    def model_name(self) -> str:
        return "convnext"

    def _create_base_model(self, input_shape: Tuple[int, int, int]) -> Model:
        """Create ConvNeXt base model pre-trained on ImageNet."""
        try:
            return tf.keras.applications.ConvNeXtBase(
                weights="imagenet",
                include_top=False,
                input_shape=input_shape,
            )
        except (AttributeError, ValueError):
            try:
                return tf.keras.applications.ConvNeXtTiny(
                    weights="imagenet",
                    include_top=False,
                    input_shape=input_shape,
                )
            except AttributeError:
                # Fallback to ResNet50V2 if ConvNeXt is not packaged in current keras
                return tf.keras.applications.ResNet50V2(
                    weights="imagenet",
                    include_top=False,
                    input_shape=input_shape,
                )

    def _get_fine_tune_layer(self) -> int:
        """Unfreeze top stage for fine-tuning."""
        return config.FINE_TUNE_LAYERS.get("convnext", -30)
