# Data subpackage — imports are available when TensorFlow is installed
# Use explicit imports: from src.data.preprocessing import preprocess_image

def __getattr__(name):
    """Lazy imports to avoid requiring TensorFlow at import time."""
    if name in ("preprocess_image", "preprocess_directory"):
        from .preprocessing import preprocess_image, preprocess_directory
        return locals()[name]
    elif name in ("get_augmentation_config", "create_augmented_generator"):
        from .augmentation import get_augmentation_config, create_augmented_generator
        return locals()[name]
    elif name in ("create_generators", "load_dataset_info", "get_sample_images"):
        from .dataset import create_generators, load_dataset_info, get_sample_images
        return locals()[name]
    raise AttributeError(f"module 'src.data' has no attribute {name!r}")
