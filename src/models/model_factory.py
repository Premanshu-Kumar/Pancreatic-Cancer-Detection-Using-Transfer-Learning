"""
Model Factory
==============
Factory pattern for creating cancer detection models by name.
"""

from typing import Optional
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from .base_model import BaseCancerModel
from .vgg16_model import VGG16CancerModel
from .resnet50_model import ResNet50CancerModel
from .inception_model import InceptionV3CancerModel


# Registry of available models
_MODEL_REGISTRY = {
    "vgg16": VGG16CancerModel,
    "resnet50": ResNet50CancerModel,
    "inceptionv3": InceptionV3CancerModel,
}


class ModelFactory:
    """
    Factory class for creating cancer detection model instances.

    Usage
    -----
    >>> model = ModelFactory.create("resnet50")
    >>> model.build()
    >>> model.compile_model()
    """

    @staticmethod
    def create(
        model_name: str,
        build: bool = False,
        compile: bool = False,
        learning_rate: Optional[float] = None,
    ) -> BaseCancerModel:
        """
        Create a cancer detection model by name.

        Parameters
        ----------
        model_name : str
            One of 'vgg16', 'resnet50', 'inceptionv3'.
        build : bool
            If True, build the model immediately.
        compile : bool
            If True, compile the model after building.
        learning_rate : float, optional
            Custom learning rate for compilation.

        Returns
        -------
        BaseCancerModel
            The instantiated model.

        Raises
        ------
        ValueError
            If model_name is not recognized.
        """
        name = model_name.lower().strip()

        if name not in _MODEL_REGISTRY:
            available = ", ".join(sorted(_MODEL_REGISTRY.keys()))
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available models: {available}"
            )

        model = _MODEL_REGISTRY[name]()

        if build or compile:
            model.build()
        if compile:
            model.compile_model(learning_rate=learning_rate)

        return model

    @staticmethod
    def list_models():
        """Return a list of available model names."""
        return list(_MODEL_REGISTRY.keys())

    @staticmethod
    def create_all(
        build: bool = False,
        compile: bool = False,
    ) -> dict:
        """
        Create instances of all available models.

        Returns
        -------
        dict
            {model_name: model_instance}
        """
        return {
            name: ModelFactory.create(name, build=build, compile=compile)
            for name in _MODEL_REGISTRY
        }
