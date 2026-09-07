"""
Grad-CAM (Gradient-weighted Class Activation Mapping)
======================================================
Generates heatmap overlays showing which image regions
influenced the model's prediction. Provides visual
explainability for the cancer detection models.

Reference:
    Selvaraju, R. R., et al. (2017).
    "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization." arXiv:1610.02391
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

matplotlib.use("Agg")

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config


class GradCAM:
    """
    Grad-CAM visualization for any CNN-based model.

    Usage
    -----
    >>> from src.utils.gradcam import GradCAM
    >>> cam = GradCAM(model.model)
    >>> heatmap, overlay = cam.generate(image_path)
    >>> cam.save_visualization(image_path, "output.png")
    """

    # Default last conv layer names per architecture
    LAST_CONV_LAYERS = {
        "vgg16": "block5_conv3",
        "resnet50": "conv5_block3_out",
        "inceptionv3": "mixed10",
    }

    def __init__(
        self,
        model: tf.keras.Model,
        last_conv_layer_name: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize Grad-CAM.

        Parameters
        ----------
        model : tf.keras.Model
            The trained Keras model.
        last_conv_layer_name : str, optional
            Name of the last convolutional layer. Auto-detected if not provided.
        model_name : str, optional
            Model identifier for auto-detecting the conv layer.
        """
        self.model = model
        self.model_name = model_name

        if last_conv_layer_name:
            self.last_conv_layer_name = last_conv_layer_name
        elif model_name and model_name in self.LAST_CONV_LAYERS:
            self.last_conv_layer_name = self.LAST_CONV_LAYERS[model_name]
        else:
            self.last_conv_layer_name = self._find_last_conv_layer()

    def _find_last_conv_layer(self) -> str:
        """Auto-detect the last convolutional layer in the model."""
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.Model):
                # Nested model (our architecture wraps base model)
                for sub_layer in reversed(layer.layers):
                    if len(sub_layer.output_shape) == 4:  # Conv layers have 4D output
                        return sub_layer.name
            if len(layer.output_shape) == 4:
                return layer.name

        raise ValueError("Could not find a convolutional layer in the model.")

    def _get_conv_layer_model(self):
        """
        Build a model or pair of models that outputs the last conv layer activations
        alongside the final predictions.
        """
        # Handle nested models (base model inside our wrapper)
        base_model = None
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model):
                base_model = layer
                break

        if base_model is not None:
            try:
                conv_layer = base_model.get_layer(self.last_conv_layer_name)
                # In Keras 3, connecting conv_layer.output directly to self.model.inputs
                # fails because conv_layer is inside base_model.
                last_conv_model = tf.keras.Model(base_model.inputs, conv_layer.output)
                classifier_input = tf.keras.Input(shape=conv_layer.output.shape[1:])
                x = classifier_input
                base_idx = self.model.layers.index(base_model)
                for layer in self.model.layers[base_idx + 1:]:
                    x = layer(x)
                classifier_model = tf.keras.Model(classifier_input, x)
                return (last_conv_model, classifier_model)
            except Exception:
                pass

        conv_layer = self.model.get_layer(self.last_conv_layer_name)
        grad_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[conv_layer.output, self.model.output],
        )

        return grad_model

    def generate(
        self,
        image_path: str,
        target_size: Tuple[int, int] = None,
        pred_index: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Generate a Grad-CAM heatmap for a given image.

        Parameters
        ----------
        image_path : str
            Path to the input image.
        target_size : tuple, optional
            (height, width) for resizing. Auto-detected from model.
        pred_index : int, optional
            Class index to generate the heatmap for.
            None = use the predicted class.

        Returns
        -------
        tuple of (heatmap, overlay, prediction)
            heatmap : np.ndarray (H, W) in [0, 1]
            overlay : np.ndarray (H, W, 3) in [0, 255]
            prediction : float — predicted probability
        """
        if target_size is None:
            input_shape = self.model.input_shape[1:3]
            target_size = input_shape

        # Load and preprocess image
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        img_array_normalized = img_array / 255.0
        img_tensor = np.expand_dims(img_array_normalized, axis=0)

        # Build gradient model
        grad_model = self._get_conv_layer_model()

        # Compute gradients
        with tf.GradientTape() as tape:
            if isinstance(grad_model, tuple):
                last_conv_model, classifier_model = grad_model
                conv_outputs = last_conv_model(img_tensor)
                tape.watch(conv_outputs)
                predictions = classifier_model(conv_outputs)
            else:
                conv_outputs, predictions = grad_model(img_tensor)

            if pred_index is None:
                pred_value = predictions[0][0]  # Binary
            else:
                pred_value = predictions[0][pred_index]

        # Get the gradients of the predicted class w.r.t. the conv layer
        grads = tape.gradient(pred_value, conv_outputs)

        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight the conv outputs by the pooled gradients
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # ReLU and normalize
        heatmap = tf.nn.relu(heatmap)
        heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        # Create overlay
        overlay = self._create_overlay(img_array, heatmap)

        prediction = float(predictions[0][0])

        return heatmap, overlay, prediction

    def _create_overlay(
        self,
        original_img: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Superimpose the heatmap on the original image.

        Parameters
        ----------
        original_img : np.ndarray
            Original image (H, W, 3) in [0, 255].
        heatmap : np.ndarray
            Grad-CAM heatmap (h, w) in [0, 1].
        alpha : float
            Blending factor for the overlay.
        colormap : int
            OpenCV colormap to apply.

        Returns
        -------
        np.ndarray
            Superimposed image (H, W, 3) in uint8.
        """
        h, w = original_img.shape[:2]

        # Resize heatmap to match original image
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Convert to uint8 and apply colormap
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Blend
        original_uint8 = original_img.astype(np.uint8)
        overlay = cv2.addWeighted(original_uint8, 1 - alpha, heatmap_colored, alpha, 0)

        return overlay

    def save_visualization(
        self,
        image_path: str,
        output_path: str = None,
        target_size: Tuple[int, int] = None,
        figsize: Tuple[int, int] = (15, 5),
    ) -> str:
        """
        Generate and save a complete Grad-CAM visualization.

        Creates a figure with three panels:
        [Original Image] [Heatmap] [Overlay]

        Parameters
        ----------
        image_path : str
            Path to the input image.
        output_path : str, optional
            Where to save the visualization.
        target_size : tuple, optional
            Image resize dimensions.
        figsize : tuple
            Figure size.

        Returns
        -------
        str
            Path to the saved visualization.
        """
        heatmap, overlay, prediction = self.generate(image_path, target_size)

        # Load original for display
        img = load_img(image_path, target_size=target_size or self.model.input_shape[1:3])
        img_array = img_to_array(img).astype(np.uint8)

        # Determine class
        pred_class = "Cancerous" if prediction >= 0.5 else "Normal"
        confidence = prediction if prediction >= 0.5 else (1 - prediction)

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Original
        axes[0].imshow(img_array)
        axes[0].set_title("Original Image", fontsize=13, fontweight="bold")
        axes[0].axis("off")

        # Heatmap
        axes[1].imshow(heatmap, cmap="jet", aspect="auto")
        axes[1].set_title("Grad-CAM Heatmap", fontsize=13, fontweight="bold")
        axes[1].axis("off")

        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay", fontsize=13, fontweight="bold")
        axes[2].axis("off")

        color = "#00b894" if pred_class == "Normal" else "#e17055"
        fig.suptitle(
            f"Prediction: {pred_class} ({confidence * 100:.1f}% confidence)",
            fontsize=15,
            fontweight="bold",
            color=color,
            y=1.02,
        )

        fig.tight_layout()

        if output_path is None:
            save_dir = config.RESULTS_DIR / "gradcam"
            save_dir.mkdir(parents=True, exist_ok=True)
            stem = Path(image_path).stem
            output_path = str(save_dir / f"gradcam_{stem}.png")

        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

        return output_path
