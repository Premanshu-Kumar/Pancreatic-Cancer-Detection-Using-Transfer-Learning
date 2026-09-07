"""
InceptionV3 Transfer Learning Model
======================================
Fine-tunes InceptionV3 (pre-trained on ImageNet) for binary
cancer detection on medical images.
"""

from typing import Tuple
from pathlib import Path

from tensorflow.keras import Model
from tensorflow.keras.applications import InceptionV3

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

from .base_model import BaseCancerModel


class InceptionV3CancerModel(BaseCancerModel):
    """
    InceptionV3-based cancer detection model.

    Architecture notes:
        - Uses Inception modules with multiple parallel filter sizes
        - Input: 299 × 299 × 3  (larger than VGG16/ResNet50)
        - ~23.9M parameters (base)
        - Captures multi-scale features effectively
        - Fine-tuning: unfreeze from mixed7 (layer 249) onwards
    """

    @property
    def model_name(self) -> str:
        return "inceptionv3"

    def _create_base_model(self, input_shape: Tuple[int, int, int]) -> Model:
        """Create InceptionV3 base model pre-trained on ImageNet."""
        return InceptionV3(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
        )

    def _get_fine_tune_layer(self) -> int:
        """
        Unfreeze from mixed7 (layer 249) onwards.

        InceptionV3 key blocks:
            mixed0 → mixed1 → ... → mixed7 → mixed8 → mixed9 → mixed10
        """
        return config.FINE_TUNE_LAYERS.get("inceptionv3", 249)
