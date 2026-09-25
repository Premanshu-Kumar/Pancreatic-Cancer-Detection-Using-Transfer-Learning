"""
Custom Loss Functions for Medical Image Classification
======================================================
Implements Binary Focal Loss and class-imbalance compensated loss formulations
to focus gradient updates on hard, ambiguous pancreatic lesions and suppress
easy negative background CT slices.
"""

from typing import Optional
import tensorflow as tf
from tensorflow.keras.losses import Loss


@tf.keras.utils.register_keras_serializable(package="PancreaticCancer")
class BinaryFocalLoss(Loss):
    """
    Binary Focal Loss:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Down-weights easy examples to focus learning on hard negative/positive boundaries.

    Parameters
    ----------
    gamma : float
        Focusing parameter. Higher values reduce relative loss for well-classified examples.
        Defaults to 2.0.
    alpha : float
        Balancing factor for the positive (cancerous) class in [0, 1]. Defaults to 0.25.
    from_logits : bool
        Whether predictions are unscaled logits or probabilities in [0, 1].
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        from_logits: bool = False,
        name: str = "binary_focal_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.from_logits = from_logits

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)

        # Clip for numerical stability
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Calculate p_t
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)

        # Calculate alpha_t
        alpha_t = y_true * self.alpha + (1.0 - y_true) * (1.0 - self.alpha)

        # Modulating factor
        focal_weight = alpha_t * tf.pow(1.0 - p_t, self.gamma)

        # Cross entropy
        bce = -tf.math.log(p_t)

        loss = focal_weight * bce
        return tf.reduce_mean(loss)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
            "from_logits": self.from_logits,
        })
        return config


def get_loss_function(
    loss_name: str = "binary_crossentropy",
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> tf.keras.losses.Loss:
    """
    Factory function to retrieve configured loss function.
    """
    name = loss_name.lower().strip()
    if name in ["focal", "focal_loss", "binary_focal_loss"]:
        return BinaryFocalLoss(gamma=gamma, alpha=alpha)
    return tf.keras.losses.BinaryCrossentropy()
