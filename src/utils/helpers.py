"""
Utility Helpers
================
Common utility functions used across the project.
"""

import os
import random
import time
from pathlib import Path

import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config


def set_seed(seed: int = None) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and TensorFlow.

    Parameters
    ----------
    seed : int, optional
        Random seed. Defaults to config.RANDOM_SEED.
    """
    if seed is None:
        seed = config.RANDOM_SEED

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    print(f"  ✓ Random seed set to {seed}")


def get_device_info() -> dict:
    """
    Get information about available compute devices.

    Returns
    -------
    dict
        Keys: 'gpus', 'gpu_names', 'cpu_count', 'tf_version'.
    """
    info = {"gpus": 0, "gpu_names": [], "cpu_count": os.cpu_count()}

    try:
        import tensorflow as tf
        info["tf_version"] = tf.__version__

        gpus = tf.config.list_physical_devices("GPU")
        info["gpus"] = len(gpus)
        info["gpu_names"] = [gpu.name for gpu in gpus]

        if gpus:
            # Enable memory growth to avoid allocating all GPU memory
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass
    except ImportError:
        info["tf_version"] = "Not installed"

    return info


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable string.

    Parameters
    ----------
    seconds : float
        Duration in seconds.

    Returns
    -------
    str
        Formatted string like "2h 15m 30s" or "45.3s".
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {s:.0f}s"
    else:
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        return f"{int(h)}h {int(m)}m {s:.0f}s"


def count_parameters(model) -> dict:
    """
    Count trainable and non-trainable parameters in a Keras model.

    Returns
    -------
    dict
        Keys: 'total', 'trainable', 'non_trainable'.
    """
    total = model.count_params()
    trainable = sum(
        np.prod(w.shape) for w in model.trainable_weights
    )
    non_trainable = total - trainable

    return {
        "total": int(total),
        "trainable": int(trainable),
        "non_trainable": int(non_trainable),
        "total_formatted": _format_params(total),
        "trainable_formatted": _format_params(trainable),
    }


def _format_params(n: int) -> str:
    """Format parameter count with M/K suffix."""
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)
