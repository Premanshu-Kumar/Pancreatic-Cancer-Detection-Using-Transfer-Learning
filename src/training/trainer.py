"""
Model Trainer
==============
Two-phase training pipeline:
  Phase 1 — Feature Extraction (frozen base)
  Phase 2 — Fine-Tuning (unfrozen top layers)

Supports training single models or all models sequentially.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

from src.models.model_factory import ModelFactory
from src.data.dataset import create_generators
from src.training.callbacks import get_callbacks, TrainingProgressCallback


class Trainer:
    """
    Orchestrates the two-phase training process for cancer detection models.

    Usage
    -----
    >>> trainer = Trainer(model_name="resnet50")
    >>> history = trainer.train()
    >>> trainer.save_results()
    """

    def __init__(
        self,
        model_name: str = None,
        batch_size: int = None,
        epochs_phase1: int = None,
        epochs_phase2: int = None,
        lr_phase1: float = None,
        lr_phase2: float = None,
    ):
        """
        Initialize the Trainer.

        Parameters
        ----------
        model_name : str
            One of 'vgg16', 'resnet50', 'inceptionv3'.
        batch_size : int
            Training batch size.
        epochs_phase1, epochs_phase2 : int
            Number of epochs for each training phase.
        lr_phase1, lr_phase2 : float
            Learning rates for each phase.
        """
        self.model_name = model_name or config.DEFAULT_MODEL
        self.batch_size = batch_size or config.BATCH_SIZE
        self.epochs_phase1 = epochs_phase1 or config.EPOCHS_PHASE1
        self.epochs_phase2 = epochs_phase2 or config.EPOCHS_PHASE2
        self.lr_phase1 = lr_phase1 or config.LEARNING_RATE_PHASE1
        self.lr_phase2 = lr_phase2 or config.LEARNING_RATE_PHASE2

        self.model_instance = None
        self.train_gen = None
        self.val_gen = None
        self.test_gen = None
        self.history_phase1 = None
        self.history_phase2 = None

    def setup(self) -> None:
        """Build and compile the model, create data generators."""
        print(f"\n{'='*60}")
        print(f"  Setting up: {self.model_name.upper()}")
        print(f"{'='*60}")

        # Create model
        self.model_instance = ModelFactory.create(
            self.model_name, build=True, compile=True,
            learning_rate=self.lr_phase1,
        )
        print(f"  ✓ Model built: {self.model_name}")
        print(f"  ✓ Input shape: {self.model_instance.input_shape}")

        # Create data generators
        self.train_gen, self.val_gen, self.test_gen = create_generators(
            model_name=self.model_name,
            batch_size=self.batch_size,
        )
        print(f"  ✓ Data generators created")
        print(f"    Train samples: {self.train_gen.samples}")
        print(f"    Val samples:   {self.val_gen.samples}")
        print(f"    Test samples:  {self.test_gen.samples}")
        print(f"    Class indices: {self.train_gen.class_indices}")

    def train_phase1(self) -> dict:
        """
        Phase 1: Feature Extraction.

        Train only the custom classification head while the
        pre-trained base model is completely frozen.

        Returns
        -------
        dict
            Training history.
        """
        print(f"\n{'─'*60}")
        print(f"  Phase 1: Feature Extraction (Frozen Base)")
        print(f"  Epochs: {self.epochs_phase1} | LR: {self.lr_phase1}")
        print(f"{'─'*60}")

        self.model_instance.freeze_base()
        self.model_instance.compile_model(learning_rate=self.lr_phase1)

        callbacks = get_callbacks(self.model_name, phase="phase1")
        callbacks.append(TrainingProgressCallback())

        start = time.time()
        history = self.model_instance.model.fit(
            self.train_gen,
            epochs=self.epochs_phase1,
            validation_data=self.val_gen,
            callbacks=callbacks,
            verbose=0,  # We use our custom callback
        )
        elapsed = time.time() - start

        self.history_phase1 = history.history
        print(f"\n  ✓ Phase 1 complete in {elapsed:.1f}s")
        print(f"    Best val_accuracy: {max(history.history['val_accuracy']):.4f}")

        return history.history

    def train_phase2(self) -> dict:
        """
        Phase 2: Fine-Tuning.

        Unfreeze the top layers of the base model and retrain
        with a much lower learning rate.

        Returns
        -------
        dict
            Training history.
        """
        print(f"\n{'─'*60}")
        print(f"  Phase 2: Fine-Tuning (Unfrozen Top Layers)")
        print(f"  Epochs: {self.epochs_phase2} | LR: {self.lr_phase2}")
        print(f"{'─'*60}")

        self.model_instance.unfreeze_base()
        self.model_instance.compile_model(learning_rate=self.lr_phase2)

        callbacks = get_callbacks(self.model_name, phase="phase2")
        callbacks.append(TrainingProgressCallback())

        start = time.time()
        history = self.model_instance.model.fit(
            self.train_gen,
            epochs=self.epochs_phase2,
            validation_data=self.val_gen,
            callbacks=callbacks,
            verbose=0,
        )
        elapsed = time.time() - start

        self.history_phase2 = history.history
        print(f"\n  ✓ Phase 2 complete in {elapsed:.1f}s")
        print(f"    Best val_accuracy: {max(history.history['val_accuracy']):.4f}")

        return history.history

    def train(self) -> Dict[str, Any]:
        """
        Run the complete two-phase training pipeline.

        Returns
        -------
        dict
            Combined training history from both phases.
        """
        self.setup()

        # Phase 1: Feature Extraction
        h1 = self.train_phase1()

        # Phase 2: Fine-Tuning
        h2 = self.train_phase2()

        # Save the final model
        self.model_instance.save()

        # Combine histories
        combined = {
            "phase1": h1,
            "phase2": h2,
            "model_name": self.model_name,
        }

        return combined

    def save_results(self, results: Optional[dict] = None) -> str:
        """
        Save training results (history, config) to JSON.

        Returns
        -------
        str
            Path to the saved results file.
        """
        if results is None:
            results = {
                "phase1": self.history_phase1,
                "phase2": self.history_phase2,
                "model_name": self.model_name,
            }

        results_dir = config.RESULTS_DIR
        results_dir.mkdir(parents=True, exist_ok=True)
        filepath = results_dir / f"{self.model_name}_training_history.json"

        # Convert numpy values to Python floats for JSON serialization
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serializable = json.loads(
            json.dumps(results, default=convert)
        )

        with open(filepath, "w") as f:
            json.dump(serializable, f, indent=2)

        print(f"  ✓ Results saved to {filepath}")
        return str(filepath)


def train_all_models() -> dict:
    """
    Train all available models sequentially.

    Returns
    -------
    dict
        {model_name: training_history}
    """
    all_results = {}

    for model_name in ModelFactory.list_models():
        print(f"\n{'═'*60}")
        print(f"  TRAINING: {model_name.upper()}")
        print(f"{'═'*60}")

        trainer = Trainer(model_name=model_name)
        results = trainer.train()
        trainer.save_results(results)
        all_results[model_name] = results

    return all_results


# =============================================================================
# CLI entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train cancer detection models"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.DEFAULT_MODEL,
        choices=["vgg16", "resnet50", "inceptionv3", "all"],
        help="Model to train (default: resnet50)",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs-p1", type=int, default=None)
    parser.add_argument("--epochs-p2", type=int, default=None)
    parser.add_argument("--lr-p1", type=float, default=None)
    parser.add_argument("--lr-p2", type=float, default=None)

    args = parser.parse_args()

    if args.model == "all":
        train_all_models()
    else:
        trainer = Trainer(
            model_name=args.model,
            batch_size=args.batch_size,
            epochs_phase1=args.epochs_p1,
            epochs_phase2=args.epochs_p2,
            lr_phase1=args.lr_p1,
            lr_phase2=args.lr_p2,
        )
        results = trainer.train()
        trainer.save_results(results)
