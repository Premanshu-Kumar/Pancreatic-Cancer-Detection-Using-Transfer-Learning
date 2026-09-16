"""
Evaluate Model and Generate Diagnostic Plots CLI
================================================
Evaluates trained cancer detection model on test dataset and generates
confusion matrix, ROC curve, and Precision-Recall plots.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Optimize TensorFlow runtime
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf

import config
from src.data.dataset import create_generators
from src.evaluation.metrics import evaluate_model
from src.evaluation.visualizations import (
    plot_training_history,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
)
from src.utils.model_downloader import ensure_model_exists


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Pancreatic Cancer Detection Models and Plot Metrics"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.DEFAULT_MODEL,
        choices=["resnet50", "vgg16", "inceptionv3"],
        help="Model architecture name to evaluate (default: resnet50)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Custom path to model weights .keras file",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.BATCH_SIZE,
        help="Batch size for test evaluation generator",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config.RESULTS_DIR),
        help="Directory to save plots and test metrics JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model.lower().strip()

    if args.weights:
        model_path = Path(args.weights)
    else:
        model_filename = f"{model_name}_best.keras"
        model_path = ensure_model_exists(model_filename)

    print("=" * 60)
    print(f"📊 Pancreatic Cancer Detection: Model Evaluation")
    print("=" * 60)
    print(f"Loading weights from: {model_path}...")
    model = tf.keras.models.load_model(str(model_path))

    _, _, test_gen = create_generators(model_name=model_name, batch_size=args.batch_size)

    sample_count = test_gen.samples if hasattr(test_gen, "samples") else "all"
    print(f"\nEvaluating on test dataset ({sample_count} samples)...")
    eval_result = evaluate_model(model, test_gen, verbose=True)
    metrics = eval_result["metrics"]

    # Setup plots directory
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Confusion Matrix
    cm_path = plot_confusion_matrix(
        metrics["confusion_matrix"], model_name=model_name, save_dir=str(plots_dir)
    )
    print(f"✓ Confusion matrix saved to: {cm_path}")

    # ROC Curve
    if "roc_curve" in metrics:
        roc_path = plot_roc_curve(
            fpr=metrics["roc_curve"]["fpr"],
            tpr=metrics["roc_curve"]["tpr"],
            auc_score=metrics.get("auc_roc", 0.0),
            model_name=model_name,
            save_dir=str(plots_dir),
        )
        print(f"✓ ROC curve saved to: {roc_path}")

    # Precision Recall Curve
    if "pr_curve" in metrics:
        pr_path = plot_precision_recall_curve(
            precision=metrics["pr_curve"]["precision"],
            recall=metrics["pr_curve"]["recall"],
            model_name=model_name,
            save_dir=str(plots_dir),
        )
        print(f"✓ PR curve saved to: {pr_path}")

    # Training history plot if JSON exists
    history_path = output_dir / f"{model_name}_training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            hist = json.load(f)
        combined_hist = {
            "accuracy": hist.get("phase1", {}).get("accuracy", [])
            + hist.get("phase2", {}).get("accuracy", []),
            "val_accuracy": hist.get("phase1", {}).get("val_accuracy", [])
            + hist.get("phase2", {}).get("val_accuracy", []),
            "loss": hist.get("phase1", {}).get("loss", [])
            + hist.get("phase2", {}).get("loss", []),
            "val_loss": hist.get("phase1", {}).get("val_loss", [])
            + hist.get("phase2", {}).get("val_loss", []),
        }
        hist_plot_path = plot_training_history(
            combined_hist, model_name=model_name, save_dir=str(plots_dir)
        )
        print(f"✓ Training history plot saved to: {hist_plot_path}")

    # Save metrics JSON
    metrics_to_save = {
        k: v
        for k, v in metrics.items()
        if k not in ("roc_curve", "pr_curve", "confusion_matrix")
    }
    if "confusion_matrix" in metrics:
        metrics_to_save["confusion_matrix"] = (
            metrics["confusion_matrix"].tolist()
            if isinstance(metrics["confusion_matrix"], np.ndarray)
            else metrics["confusion_matrix"]
        )

    metrics_file = output_dir / f"{model_name}_test_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_to_save, f, indent=2)

    print(f"✓ Test metrics saved to: {metrics_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
