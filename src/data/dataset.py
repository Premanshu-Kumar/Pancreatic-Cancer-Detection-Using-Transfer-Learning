"""
Dataset Loading & Generator Creation
======================================
High-level functions for creating train/val/test data generators
and inspecting dataset composition.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import csv
import numpy as np
from PIL import Image
import tensorflow as tf

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


def resolve_dataset_root(custom_path: Optional[str] = None) -> Path:
    """
    Resolve dataset directory, prioritizing custom path, then DATA_DIR from environment,
    with verification that external storage (e.g. D:, E:, /mnt/data) is accessible.
    """
    if custom_path:
        path = Path(custom_path)
    else:
        path = Path(os.environ.get("DATA_DIR", config.DATA_DIR))
    return path


def create_generators(
    model_name: str = None,
    train_dir: str = None,
    val_dir: str = None,
    test_dir: str = None,
    batch_size: int = None,
    custom_augmentation: Optional[Dict[str, Any]] = None,
    data_root: Optional[str] = None,
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
    data_root : str, optional
        Base root directory (e.g., external drive mount) for data splits.

    Returns
    -------
    tuple of (train_gen, val_gen, test_gen)
        Keras DirectoryIterators.
    """
    base_root = resolve_dataset_root(data_root)

    if model_name is None:
        model_name = config.DEFAULT_MODEL
    if train_dir is None:
        train_dir = str(base_root / "processed" / "train" if (base_root / "processed" / "train").exists() else config.TRAIN_DIR)
    if val_dir is None:
        val_dir = str(base_root / "processed" / "val" if (base_root / "processed" / "val").exists() else config.VAL_DIR)
    if test_dir is None:
        test_dir = str(base_root / "processed" / "test" if (base_root / "processed" / "test").exists() else config.TEST_DIR)
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


def parse_and_preprocess_image(
    file_path: tf.Tensor,
    label: tf.Tensor,
    target_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Read image file, decode PNG/JPEG, resize, and normalize using pure TensorFlow ops."""
    img_bytes = tf.io.read_file(file_path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, target_size)
    if normalize:
        img = tf.cast(img, tf.float32) / 255.0
    return img, label


def create_tf_dataset_from_manifest(
    manifest_csv: str,
    target_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    is_training: bool = True,
    shuffle_buffer: int = 1000,
) -> tf.data.Dataset:
    """
    Create a high-throughput tf.data.Dataset from a CSV manifest.
    Utilizes AUTOTUNE parallel calls, caching, and prefetching to eliminate I/O bottlenecks.
    """
    file_paths = []
    labels = []

    with open(manifest_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_paths.append(row["filepath"])
            labels.append(float(row["label"]))

    if not file_paths:
        raise ValueError(f"No records found in manifest: {manifest_csv}")

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(file_paths), shuffle_buffer))

    # Parallel mapping with AUTOTUNE
    dataset = dataset.map(
        lambda path, lbl: parse_and_preprocess_image(path, lbl, target_size=target_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def create_tf_data_pipeline(
    data_dir: Optional[str] = None,
    target_size: Tuple[int, int] = None,
    batch_size: int = None,
) -> Dict[str, tf.data.Dataset]:
    """
    Create high-throughput tf.data pipelines for train, validation, and test splits.
    Defaults to patient manifests in data/splits/ if available.
    """
    if target_size is None:
        target_size = config.DEFAULT_IMG_SIZE
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    splits_dir = config.DATA_DIR / "splits"
    train_csv = splits_dir / "train_manifest.csv"
    val_csv = splits_dir / "val_manifest.csv"
    test_csv = splits_dir / "test_manifest.csv"

    if train_csv.exists() and val_csv.exists() and test_csv.exists():
        return {
            "train": create_tf_dataset_from_manifest(str(train_csv), target_size, batch_size, is_training=True),
            "val": create_tf_dataset_from_manifest(str(val_csv), target_size, batch_size, is_training=False),
            "test": create_tf_dataset_from_manifest(str(test_csv), target_size, batch_size, is_training=False),
        }

    # Fallback to tf.keras.utils.image_dataset_from_directory with prefetching
    base_path = Path(data_dir) if data_dir else config.PROCESSED_DATA_DIR
    datasets = {}
    for split in ["train", "val", "test"]:
        sdir = base_path / split
        if sdir.exists():
            ds = tf.keras.utils.image_dataset_from_directory(
                str(sdir),
                image_size=target_size,
                batch_size=batch_size,
                label_mode="binary",
                shuffle=(split == "train"),
            )
            ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
            datasets[split] = ds
    return datasets

