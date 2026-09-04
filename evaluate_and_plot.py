"""
Evaluate Trained Model and Generate Plots
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from src.data.dataset import create_generators
from src.evaluation.metrics import evaluate_model
from src.evaluation.visualizations import (
    plot_training_history,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
)

model_name = sys.argv[1] if len(sys.argv) > 1 else "resnet50"
model_path = config.MODELS_DIR / f"{model_name}_best.keras"

print(f"Loading {model_path}...")
model = tf.keras.models.load_model(str(model_path))

_, _, test_gen = create_generators(model_name=model_name, batch_size=config.BATCH_SIZE)

print("\nEvaluating on test dataset (310 samples)...")
eval_result = evaluate_model(model, test_gen, verbose=True)
metrics = eval_result["metrics"]

# Generate plots
plots_dir = config.RESULTS_DIR / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

# Confusion Matrix
cm_path = plot_confusion_matrix(metrics["confusion_matrix"], model_name=model_name, save_dir=str(plots_dir))
print(f"✓ Confusion matrix saved to: {cm_path}")

# ROC Curve
roc_path = plot_roc_curve(
    fpr=metrics["roc_curve"]["fpr"],
    tpr=metrics["roc_curve"]["tpr"],
    auc_score=metrics["auc_roc"],
    model_name=model_name,
    save_dir=str(plots_dir)
)
print(f"✓ ROC curve saved to: {roc_path}")

# Precision Recall Curve
pr_path = plot_precision_recall_curve(
    precision=metrics["pr_curve"]["precision"],
    recall=metrics["pr_curve"]["recall"],
    model_name=model_name,
    save_dir=str(plots_dir)
)
print(f"✓ PR curve saved to: {pr_path}")

# Training history plot from saved history JSON
history_path = config.RESULTS_DIR / f"{model_name}_training_history.json"
if history_path.exists():
    with open(history_path) as f:
        hist = json.load(f)
    combined_hist = {
        "accuracy": hist["phase1"]["accuracy"] + hist["phase2"]["accuracy"],
        "val_accuracy": hist["phase1"]["val_accuracy"] + hist["phase2"]["val_accuracy"],
        "loss": hist["phase1"]["loss"] + hist["phase2"]["loss"],
        "val_loss": hist["phase1"]["val_loss"] + hist["phase2"]["val_loss"],
    }
    hist_plot_path = plot_training_history(combined_hist, model_name=model_name, save_dir=str(plots_dir))
    print(f"✓ Training history plot saved to: {hist_plot_path}")

# Save metrics JSON
metrics_to_save = {k: v for k, v in metrics.items() if k not in ("roc_curve", "pr_curve", "confusion_matrix")}
metrics_to_save["confusion_matrix"] = metrics["confusion_matrix"].tolist()
metrics_file = config.RESULTS_DIR / f"{model_name}_test_metrics.json"
with open(metrics_file, "w") as f:
    json.dump(metrics_to_save, f, indent=2)

print(f"✓ Test metrics saved to: {metrics_file}")
