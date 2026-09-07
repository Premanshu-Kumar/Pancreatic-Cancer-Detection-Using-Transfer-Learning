"""
Training Callbacks
===================
Custom and built-in Keras callbacks for monitoring
training, preventing overfitting, and saving checkpoints.
"""

from pathlib import Path
from typing import List, Optional

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard,
    CSVLogger,
)

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config


def get_callbacks(
    model_name: str,
    phase: str = "phase1",
    include_tensorboard: bool = False,
    custom_patience: Optional[int] = None,
) -> List[tf.keras.callbacks.Callback]:
    """
    Create a list of training callbacks.

    Parameters
    ----------
    model_name : str
        Model identifier for naming checkpoint files.
    phase : str
        Training phase ('phase1' for feature extraction, 'phase2' for fine-tuning).
    include_tensorboard : bool
        Whether to include TensorBoard callback.
    custom_patience : int, optional
        Override default early stopping patience.

    Returns
    -------
    list of keras Callbacks
        Configured callbacks ready for model.fit().
    """
    patience = custom_patience or config.EARLY_STOPPING_PATIENCE

    callbacks = []

    # 1. Early Stopping — halt training if val_loss stops improving
    callbacks.append(
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            verbose=1,
            restore_best_weights=True,
            mode="min",
        )
    )

    # 2. Model Checkpoint — save the best model based on val_accuracy
    checkpoint_path = str(
        config.MODELS_DIR / f"{model_name}_{phase}_best.keras"
    )
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    callbacks.append(
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_accuracy",
            verbose=1,
            save_best_only=True,
            mode="max",
        )
    )

    # 3. Reduce Learning Rate — lower LR when validation plateaus
    callbacks.append(
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            verbose=1,
            min_lr=config.MIN_LEARNING_RATE,
            mode="min",
        )
    )

    # 4. CSV Logger — log epoch metrics to a CSV file
    log_dir = config.RESULTS_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    callbacks.append(
        CSVLogger(
            str(log_dir / f"{model_name}_{phase}_training.csv"),
            separator=",",
            append=False,
        )
    )

    # 5. TensorBoard (optional)
    if include_tensorboard:
        tb_dir = config.RESULTS_DIR / "tensorboard" / f"{model_name}_{phase}"
        tb_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            TensorBoard(
                log_dir=str(tb_dir),
                histogram_freq=1,
                write_graph=True,
            )
        )

    return callbacks


class TrainingProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback that prints a clean progress summary per epoch."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        metrics = []
        for key in ["loss", "accuracy", "val_loss", "val_accuracy"]:
            if key in logs:
                label = key.replace("_", " ").title()
                metrics.append(f"{label}: {logs[key]:.4f}")

        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        metrics.append(f"LR: {lr:.2e}")

        print(f"  Epoch {epoch + 1}: {' | '.join(metrics)}")
