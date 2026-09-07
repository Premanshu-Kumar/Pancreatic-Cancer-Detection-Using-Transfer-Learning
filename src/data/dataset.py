"""
Dataset Loading & Generator Creation
======================================
High-level functions for creating train/val/test data generators
and inspecting dataset composition.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
from PIL import Image

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

from .augmentation import create_augmented_generator


def _get_preprocessing_function(model_name: str):
    """
    Dynamically import the model-specific preprocessing function.

    Parameters
    ----------
    model_name : str
        One of 'vgg16', 'resnet50', 'inceptionv3'.

    Returns
    -------
    callable or None
        The preprocessing function, or None if not found.
    """
    model_name = model_name.lower()
    try:
        if model_name == "vgg16":
            from tensorflow.keras.applications.vgg16 import preprocess_input
        elif model_name == "resnet50":
            from tensorflow.keras.applications.resnet50 import preprocess_input
        elif model_name == "inceptionv3":
            from tensorflow.keras.applications.inception_v3 import preprocess_input
        else:
            return None
        return preprocess_input
    except ImportError:
        return None


def create_generators(
    model_name: str = None,
    train_dir: str = None,
    val_dir: str = None,
    test_dir: str = None,
    batch_size: int = None,
    custom_augmentation: Optional[Dict[str, Any]] = None,
) -> Tuple:
    """
    Create train, validation, and test data generators.

    Parameters
    ----------
    model_name : str, optional
        Model name for preprocessing and image size. Defaults to config.DEFAULT_MODEL.
    train_dir, val_dir, test_dir : str, optional
        Directories for each split. Defaults from config.
    batch_size : int, optional
        Batch size. Defaults from config.
    custom_augmentation : dict, optional
        Override augmentation parameters.

    Returns
    -------
    tuple of (train_gen, val_gen, test_gen)
        Keras DirectoryIterators.
    """
    if model_name is None:
        model_name = config.DEFAULT_MODEL
    if train_dir is None:
        train_dir = str(config.TRAIN_DIR)
    if val_dir is None:
        val_dir = str(config.VAL_DIR)
    if test_dir is None:
        test_dir = str(config.TEST_DIR)
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    model_config = config.IMAGE_CONFIGS.get(model_name.lower(), {})
    target_size = model_config.get("img_size", config.DEFAULT_IMG_SIZE)
    preprocess_fn = _get_preprocessing_function(model_name)

    # Training generator (with augmentation)
    train_gen = create_augmented_generator(
        directory=train_dir,
        target_size=target_size,
        batch_size=batch_size,
        preprocessing_function=preprocess_fn,
        is_training=True,
        shuffle=True,
        custom_augmentation=custom_augmentation,
    )

    # Validation generator (no augmentation)
    val_gen = create_augmented_generator(
        directory=val_dir,
        target_size=target_size,
        batch_size=batch_size,
        preprocessing_function=preprocess_fn,
        is_training=False,
        shuffle=False,
    )

    # Test generator (no augmentation, no shuffle)
    test_gen = create_augmented_generator(
        directory=test_dir,
        target_size=target_size,
        batch_size=batch_size,
        preprocessing_function=preprocess_fn,
        is_training=False,
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


def load_dataset_info(data_dir: str = None) -> dict:
    """
    Scan the dataset directory and return class distribution info.

    Parameters
    ----------
    data_dir : str, optional
        Root data directory containing train/val/test splits.
        Defaults to config.PROCESSED_DATA_DIR.

    Returns
    -------
    dict
        Nested dict with split → class → count mappings.
        Example: {'train': {'Normal': 500, 'Cancerous': 480}, ...}
    """
    if data_dir is None:
        data_dir = str(config.PROCESSED_DATA_DIR)

    data_path = Path(data_dir)
    info = {}

    for split in ["train", "val", "test"]:
        split_dir = data_path / split
        if not split_dir.exists():
            info[split] = {}
            continue

        split_info = {}
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.is_dir():
                valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
                count = sum(
                    1 for f in class_dir.iterdir()
                    if f.suffix.lower() in valid_ext
                )
                split_info[class_dir.name] = count

        info[split] = split_info

    # Summary
    total = sum(
        count for split_data in info.values()
        for count in split_data.values()
    )
    info["total"] = total

    return info


def get_sample_images(
    data_dir: str = None,
    n_samples: int = 5,
    split: str = "train",
    seed: int = None,
) -> Dict[str, List[np.ndarray]]:
    """
    Load a few sample images per class for visualization.

    Parameters
    ----------
    data_dir : str, optional
        Root data directory. Defaults to config.PROCESSED_DATA_DIR.
    n_samples : int
        Number of samples per class.
    split : str
        Which split to sample from ('train', 'val', 'test').
    seed : int, optional
        Random seed.

    Returns
    -------
    dict
        {class_name: [list of np.ndarray images]}
    """
    if data_dir is None:
        data_dir = str(config.PROCESSED_DATA_DIR)
    if seed is None:
        seed = config.RANDOM_SEED

    rng = np.random.RandomState(seed)
    split_dir = Path(data_dir) / split
    samples = {}

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        all_images = [
            f for f in class_dir.iterdir()
            if f.suffix.lower() in valid_ext
        ]

        if len(all_images) == 0:
            samples[class_dir.name] = []
            continue

        chosen = rng.choice(
            len(all_images),
            size=min(n_samples, len(all_images)),
            replace=False,
        )

        class_samples = []
        for idx in chosen:
            img = Image.open(all_images[idx]).convert("RGB")
            class_samples.append(np.array(img))

        samples[class_dir.name] = class_samples

    return samples
