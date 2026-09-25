"""
Data Augmentation Strategies
=============================
Configures image augmentation using Keras ImageDataGenerator
to improve model generalization and combat overfitting.
"""

from typing import Dict, Any, Optional, Tuple
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config


import cv2
import numpy as np

try:
    import albumentations as A
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


def get_medical_transforms(
    img_size: Optional[Tuple[int, int]] = None,
    is_training: bool = True,
    p_flip: float = 0.5,
    p_contrast: float = 0.3,
):
    """
    Build a clinically sound Albumentations pipeline for abdominal CT scans.
    Enforces medical imaging constraints:
    - Retains anatomical axis orientation (avoids unrealistic vertical flipping).
    - Uses subtle affine translation (max 6%) and gentle rotation (+-15 deg or 90 deg).
    - Preserves CT HU intensity distributions by avoiding severe artificial color distortions.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        return None

    if not is_training:
        transforms = []
        if img_size:
            transforms.append(A.Resize(height=img_size[0], width=img_size[1]))
        return A.Compose(transforms)

    aug_list = []
    if img_size:
        aug_list.append(A.Resize(height=img_size[0], width=img_size[1]))

    # Safe anatomical transforms
    aug_list.extend([
        A.HorizontalFlip(p=p_flip),
        A.RandomRotate90(p=0.4),
        A.ShiftScaleRotate(
            shift_limit=0.06,
            scale_limit=0.08,
            rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.10,
            contrast_limit=0.15,
            p=p_contrast,
        ),
    ])
    return A.Compose(aug_list)


def apply_medical_augmentation(image: np.ndarray, transform=None) -> np.ndarray:
    """
    Apply medical augmentation pipeline to a single image.
    """
    if transform is None:
        transform = get_medical_transforms(is_training=True)

    if transform is not None:
        augmented = transform(image=image)
        return augmented["image"]

    # Fallback NumPy subtle augmentations if albumentations is unavailable
    img = image.copy()
    if np.random.rand() > 0.5:
        img = np.fliplr(img)
    return img


def get_augmentation_config(custom_config: Optional[Dict[str, Any]] = None) -> dict:
    """
    Get the data augmentation configuration with medical-safe constraints.
    """
    aug_config = config.AUGMENTATION_CONFIG.copy()
    if custom_config:
        aug_config.update(custom_config)
    # Medical safety overrides: prevent inverted anatomy and excessive rotation
    aug_config["vertical_flip"] = False
    aug_config["rotation_range"] = min(aug_config.get("rotation_range", 15), 15)
    aug_config["zoom_range"] = min(aug_config.get("zoom_range", 0.1), 0.1)
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
