"""
Abstract Base Model for Cancer Detection
==========================================
Defines the common interface and shared logic for all
transfer learning model implementations.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config


class BaseCancerModel(ABC):
    """
    Abstract base class for transfer learning cancer detection models.

    Subclasses must implement:
        - _create_base_model()
        - _get_fine_tune_layer()
        - model_name (property)
    """

    def __init__(self):
        self.model: Optional[Model] = None
        self.base_model: Optional[Model] = None
        self.history = None
        self._is_built = False

    # ------------------------------------------------------------------
    # Abstract methods (must be implemented by subclasses)
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier string (e.g., 'vgg16')."""
        pass

    @abstractmethod
    def _create_base_model(self, input_shape: Tuple[int, int, int]) -> Model:
        """
        Create and return the pre-trained base model (without top layers).

        Parameters
        ----------
        input_shape : tuple
            (height, width, channels)

        Returns
        -------
        tf.keras.Model
            Pre-trained base model.
        """
        pass

    @abstractmethod
    def _get_fine_tune_layer(self) -> int:
        """
        Return the layer index from which to unfreeze for fine-tuning.

        Returns
        -------
        int
            Layer index. Negative values count from the end.
        """
        pass

    # ------------------------------------------------------------------
    # Common implementation
    # ------------------------------------------------------------------

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        """Get the input shape for this model from config."""
        model_cfg = config.IMAGE_CONFIGS.get(self.model_name, {})
        return model_cfg.get("input_shape", config.DEFAULT_INPUT_SHAPE)

    @property
    def img_size(self) -> Tuple[int, int]:
        """Get the image size (height, width) for this model."""
        return self.input_shape[:2]

    def build(self) -> Model:
        """
        Build the complete model architecture.

        Architecture:
            Pre-trained Base → GlobalAveragePooling2D → Dense(512) →
            BatchNorm → Dropout(0.5) → Dense(256) → BatchNorm →
            Dropout(0.3) → Dense(1, sigmoid)

        Returns
        -------
        tf.keras.Model
            The built model.
        """
        # Create base model
        self.base_model = self._create_base_model(self.input_shape)
        self.base_model.trainable = False  # Freeze for feature extraction

        # Build classification head
        inputs = tf.keras.Input(shape=self.input_shape)
        x = self.base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(512, activation="relu", name="fc_512")(x)
        x = layers.BatchNormalization(name="bn_512")(x)
        x = layers.Dropout(0.5, name="dropout_05")(x)

        x = layers.Dense(256, activation="relu", name="fc_256")(x)
        x = layers.BatchNormalization(name="bn_256")(x)
        x = layers.Dropout(0.3, name="dropout_03")(x)

        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

        self.model = Model(inputs, outputs, name=f"{self.model_name}_cancer_detector")
        self._is_built = True

        return self.model

    def compile_model(
        self,
        learning_rate: float = None,
        optimizer: str = None,
    ) -> None:
        """
        Compile the model with optimizer, loss, and metrics.

        Parameters
        ----------
        learning_rate : float, optional
            Learning rate. Defaults to config.LEARNING_RATE_PHASE1.
        optimizer : str, optional
            Optimizer name. Defaults to config.OPTIMIZER.
        """
        if not self._is_built:
            self.build()

        if learning_rate is None:
            learning_rate = config.LEARNING_RATE_PHASE1

        opt = optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(
            optimizer=opt,
            loss=config.LOSS_FUNCTION,
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )

    def freeze_base(self) -> None:
        """Freeze all layers in the base model."""
        if self.base_model is not None:
            self.base_model.trainable = False
            print(f"  ✓ {self.model_name}: Base model frozen")

    def unfreeze_base(self, from_layer: int = None) -> None:
        """
        Unfreeze layers in the base model for fine-tuning.

        Parameters
        ----------
        from_layer : int, optional
            Layer index from which to unfreeze.
            If negative, counts from the end.
            Defaults to _get_fine_tune_layer().
        """
        if self.base_model is None:
            raise ValueError("Model not built yet. Call build() first.")

        if from_layer is None:
            from_layer = self._get_fine_tune_layer()

        self.base_model.trainable = True

        if from_layer < 0:
            # Negative indexing: freeze all except last N layers
            for layer in self.base_model.layers[:from_layer]:
                layer.trainable = False
        else:
            # Positive indexing: freeze layers before index
            for layer in self.base_model.layers[:from_layer]:
                layer.trainable = False

        trainable_count = sum(
            1 for layer in self.base_model.layers if layer.trainable
        )
        total_count = len(self.base_model.layers)
        print(
            f"  ✓ {self.model_name}: Unfrozen {trainable_count}/{total_count} "
            f"base layers for fine-tuning"
        )

    def get_summary(self) -> str:
        """Print and return the model summary."""
        if self.model is None:
            self.build()
        self.model.summary()
        # Also return as string
        lines = []
        self.model.summary(print_fn=lambda x: lines.append(x))
        return "\n".join(lines)

    def save(self, filepath: str = None) -> str:
        """
        Save the trained model to disk.

        Parameters
        ----------
        filepath : str, optional
            Full path for the saved model. Defaults to
            models/{model_name}_best.keras.

        Returns
        -------
        str
            Path where the model was saved.
        """
        if filepath is None:
            filepath = str(config.MODELS_DIR / f"{self.model_name}_best.keras")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)
        print(f"  ✓ Model saved to {filepath}")
        return filepath

    def load(self, filepath: str = None) -> None:
        """
        Load a trained model from disk.

        Parameters
        ----------
        filepath : str, optional
            Path to the saved model file.
        """
        if filepath is None:
            filepath = str(config.MODELS_DIR / f"{self.model_name}_best.keras")

        self.model = tf.keras.models.load_model(filepath)
        self._is_built = True
        print(f"  ✓ Model loaded from {filepath}")

    def predict(
        self,
        image_path: str,
        threshold: float = 0.5,
    ) -> dict:
        """
        Predict whether a single image is Normal or Cancerous.

        Parameters
        ----------
        image_path : str
            Path to the input image.
        threshold : float
            Classification threshold. Above = Cancerous, Below = Normal.

        Returns
        -------
        dict
            Keys: 'class', 'confidence', 'probability', 'label_index'.
        """
        if self.model is None:
            raise ValueError("No model loaded. Call build() or load() first.")

        # Load and preprocess the image
        img = load_img(image_path, target_size=self.img_size)
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict
        probability = float(self.model.predict(img_array, verbose=0)[0][0])

        if probability >= threshold:
            predicted_class = "Cancerous"
            confidence = probability
            label_index = 1
        else:
            predicted_class = "Normal"
            confidence = 1.0 - probability
            label_index = 0

        return {
            "class": predicted_class,
            "confidence": round(confidence * 100, 2),
            "probability": round(probability, 4),
            "label_index": label_index,
        }

    def predict_batch(
        self,
        image_paths: List[str],
        threshold: float = 0.5,
    ) -> List[dict]:
        """Predict on multiple images."""
        return [self.predict(p, threshold) for p in image_paths]
