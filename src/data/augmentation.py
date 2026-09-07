"""
Data Augmentation Strategies
=============================
Configures image augmentation using Keras ImageDataGenerator
to improve model generalization and combat overfitting.
"""

from typing import Dict, Any, Optional
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config


def get_augmentation_config(custom_config: Optional[Dict[str, Any]] = None) -> dict:
    """
    Get the data augmentation configuration.

    Parameters
    ----------
    custom_config : dict, optional
        Override default augmentation parameters.

    Returns
    -------
    dict
        Augmentation configuration dictionary.
    """
    aug_config = config.AUGMENTATION_CONFIG.copy()
    if custom_config:
        aug_config.update(custom_config)
    return aug_config


def create_train_generator(
    preprocessing_function=None,
    custom_augmentation: Optional[Dict[str, Any]] = None,
) -> ImageDataGenerator:
    """
    Create an ImageDataGenerator with augmentation for training data.

    Parameters
    ----------
    preprocessing_function : callable, optional
        Model-specific preprocessing function (e.g., VGG16's preprocess_input).
    custom_augmentation : dict, optional
        Override default augmentation parameters.

    Returns
    -------
    ImageDataGenerator
        Configured generator with augmentation.
    """
    aug_config = get_augmentation_config(custom_augmentation)

    return ImageDataGenerator(
        rescale=1.0 / 255.0 if preprocessing_function is None else None,
        preprocessing_function=preprocessing_function,
        rotation_range=aug_config["rotation_range"],
        width_shift_range=aug_config["width_shift_range"],
        height_shift_range=aug_config["height_shift_range"],
        shear_range=aug_config["shear_range"],
        zoom_range=aug_config["zoom_range"],
        horizontal_flip=aug_config["horizontal_flip"],
        vertical_flip=aug_config["vertical_flip"],
        brightness_range=aug_config.get("brightness_range"),
        fill_mode=aug_config["fill_mode"],
    )


def create_val_test_generator(
    preprocessing_function=None,
) -> ImageDataGenerator:
    """
    Create an ImageDataGenerator for validation/test data (no augmentation).

    Only rescaling/preprocessing is applied — no geometric transforms.

    Parameters
    ----------
    preprocessing_function : callable, optional
        Model-specific preprocessing function.

    Returns
    -------
    ImageDataGenerator
        Generator with only rescaling.
    """
    return ImageDataGenerator(
        rescale=1.0 / 255.0 if preprocessing_function is None else None,
        preprocessing_function=preprocessing_function,
    )


def create_augmented_generator(
    directory: str,
    target_size=None,
    batch_size: int = None,
    class_mode: str = None,
    preprocessing_function=None,
    is_training: bool = True,
    shuffle: bool = True,
    seed: int = None,
    custom_augmentation: Optional[Dict[str, Any]] = None,
):
    """
    Create a directory-based data generator with optional augmentation.

    Parameters
    ----------
    directory : str
        Path to the data directory (with class subfolders).
    target_size : tuple, optional
        (height, width) for images. Defaults from config.
    batch_size : int, optional
        Batch size. Defaults from config.
    class_mode : str, optional
        'binary' or 'categorical'. Defaults from config.
    preprocessing_function : callable, optional
        Model-specific preprocessing.
    is_training : bool
        If True, apply augmentation. If False, only rescale.
    shuffle : bool
        Whether to shuffle the data.
    seed : int, optional
        Random seed.
    custom_augmentation : dict, optional
        Override augmentation parameters.

    Returns
    -------
    DirectoryIterator
        Keras directory iterator yielding batches of (images, labels).
    """
    if target_size is None:
        target_size = config.DEFAULT_IMG_SIZE
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if class_mode is None:
        class_mode = config.CLASS_MODE
    if seed is None:
        seed = config.RANDOM_SEED

    if is_training:
        datagen = create_train_generator(preprocessing_function, custom_augmentation)
    else:
        datagen = create_val_test_generator(preprocessing_function)

    return datagen.flow_from_directory(
        directory,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=shuffle,
        seed=seed,
    )
