"""
Evaluation Visualizations
===========================
Publication-quality plots for training history, confusion matrix,
ROC curves, and model comparison charts.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.use("Agg")  # Non-interactive backend for saving plots

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config

# Global style settings
plt.rcParams.update({
    "figure.facecolor": "#0f0f23",
    "axes.facecolor": "#1a1a3e",
    "axes.edgecolor": "#444477",
    "axes.labelcolor": "#ccccee",
    "text.color": "#ccccee",
    "xtick.color": "#9999bb",
    "ytick.color": "#9999bb",
    "grid.color": "#333366",
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.size": 11,
})

# Color palette
COLORS = {
    "primary": "#6c5ce7",
    "secondary": "#00cec9",
    "accent": "#fd79a8",
    "success": "#00b894",
    "warning": "#fdcb6e",
    "danger": "#e17055",
    "vgg16": "#6c5ce7",
    "resnet50": "#00cec9",
    "inceptionv3": "#fd79a8",
}


def _save_figure(fig, filename: str, save_dir: str = None) -> str:
    """Save a figure to the results directory."""
    if save_dir is None:
        save_dir = str(config.RESULTS_DIR / "plots")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filepath = str(Path(save_dir) / filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return filepath


def plot_training_history(
    history: dict,
    model_name: str = "model",
    save_dir: str = None,
) -> str:
    """
    Plot training & validation accuracy and loss curves.

    Parameters
    ----------
    history : dict
        Keras training history dictionary.
    model_name : str
        Model name for the title and filename.
    save_dir : str, optional
        Directory to save the plot.

    Returns
    -------
    str
        Path to the saved figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["accuracy"]) + 1)

    # Accuracy
    axes[0].plot(epochs, history["accuracy"], color=COLORS["primary"],
                 linewidth=2, label="Train Accuracy", marker="o", markersize=3)
    axes[0].plot(epochs, history["val_accuracy"], color=COLORS["secondary"],
                 linewidth=2, label="Val Accuracy", marker="s", markersize=3)
    axes[0].set_title(f"{model_name.upper()} — Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(facecolor="#1a1a3e", edgecolor="#444477")
    axes[0].grid(True)

    # Loss
    axes[1].plot(epochs, history["loss"], color=COLORS["accent"],
                 linewidth=2, label="Train Loss", marker="o", markersize=3)
    axes[1].plot(epochs, history["val_loss"], color=COLORS["warning"],
                 linewidth=2, label="Val Loss", marker="s", markersize=3)
    axes[1].set_title(f"{model_name.upper()} — Loss", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(facecolor="#1a1a3e", edgecolor="#444477")
    axes[1].grid(True)

    fig.suptitle(f"Training History — {model_name.upper()}",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()

    return _save_figure(fig, f"{model_name}_training_history.png", save_dir)


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str = "model",
    class_names: list = None,
    save_dir: str = None,
) -> str:
    """
    Plot a confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray
        2×2 confusion matrix.
    model_name : str
        Model name for title and filename.
    class_names : list, optional
        Class labels. Defaults to config.CLASS_NAMES.
    save_dir : str, optional
        Save directory.

    Returns
    -------
    str
        Path to the saved figure.
    """
    if class_names is None:
        class_names = config.CLASS_NAMES

    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=2,
        linecolor="#1a1a3e",
        annot_kws={"size": 18, "fontweight": "bold"},
        ax=ax,
    )

    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)
    ax.set_title(f"Confusion Matrix — {model_name.upper()}",
                 fontsize=15, fontweight="bold", pad=15)

    fig.tight_layout()
    return _save_figure(fig, f"{model_name}_confusion_matrix.png", save_dir)


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    model_name: str = "model",
    save_dir: str = None,
) -> str:
    """
    Plot the ROC curve.

    Parameters
    ----------
    fpr, tpr : np.ndarray
        False positive rate and true positive rate.
    auc_score : float
        Area under the ROC curve.
    model_name : str
        Model name.
    save_dir : str, optional
        Save directory.

    Returns
    -------
    str
        Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(fpr, tpr, color=COLORS["primary"], linewidth=2.5,
            label=f"ROC (AUC = {auc_score:.4f})")
    ax.plot([0, 1], [0, 1], color=COLORS["danger"], linewidth=1,
            linestyle="--", alpha=0.7, label="Random (AUC = 0.50)")

    ax.fill_between(fpr, tpr, alpha=0.15, color=COLORS["primary"])

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title(f"ROC Curve — {model_name.upper()}",
                 fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", facecolor="#1a1a3e", edgecolor="#444477",
              fontsize=11)
    ax.grid(True)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    fig.tight_layout()
    return _save_figure(fig, f"{model_name}_roc_curve.png", save_dir)


def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    model_name: str = "model",
    save_dir: str = None,
) -> str:
    """
    Plot the Precision-Recall curve.

    Returns
    -------
    str
        Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(recall, precision, color=COLORS["secondary"], linewidth=2.5,
            label="Precision-Recall")
    ax.fill_between(recall, precision, alpha=0.15, color=COLORS["secondary"])

    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title(f"Precision-Recall Curve — {model_name.upper()}",
                 fontsize=15, fontweight="bold")
    ax.legend(loc="lower left", facecolor="#1a1a3e", edgecolor="#444477")
    ax.grid(True)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    fig.tight_layout()
    return _save_figure(fig, f"{model_name}_pr_curve.png", save_dir)


def plot_sample_predictions(
    images: List[np.ndarray],
    true_labels: List[int],
    pred_labels: List[int],
    confidences: List[float],
    model_name: str = "model",
    n_cols: int = 5,
    save_dir: str = None,
) -> str:
    """
    Plot a grid of sample predictions with true/predicted labels.

    Returns
    -------
    str
        Path to the saved figure.
    """
    n = len(images)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.5 * n_rows))

    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
        axes = np.array(axes).reshape(1, -1)

    class_names = config.CLASS_NAMES

    for i in range(n_rows * n_cols):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]

        if i < n:
            img = images[i]
            if img.max() > 1.0:
                img = img / 255.0
            ax.imshow(img)

            true_name = class_names[true_labels[i]]
            pred_name = class_names[pred_labels[i]]
            conf = confidences[i]

            is_correct = true_labels[i] == pred_labels[i]
            color = COLORS["success"] if is_correct else COLORS["danger"]

            ax.set_title(
                f"True: {true_name}\nPred: {pred_name} ({conf:.1f}%)",
                fontsize=9,
                color=color,
                fontweight="bold",
            )
        ax.axis("off")

    fig.suptitle(f"Sample Predictions — {model_name.upper()}",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save_figure(fig, f"{model_name}_sample_predictions.png", save_dir)


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    save_dir: str = None,
) -> str:
    """
    Plot a bar chart comparing metrics across all models.

    Parameters
    ----------
    results : dict
        {model_name: {metric_name: value, ...}}
        Example: {'vgg16': {'accuracy': 0.92, 'precision': 0.90, ...}}
    save_dir : str, optional
        Save directory.

    Returns
    -------
    str
        Path to the saved figure.
    """
    metrics_to_compare = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    model_names = list(results.keys())

    x = np.arange(len(metrics_to_compare))
    width = 0.8 / len(model_names)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model_name in enumerate(model_names):
        values = [
            results[model_name].get(m, 0) for m in metrics_to_compare
        ]
        color = COLORS.get(model_name, COLORS["primary"])
        bars = ax.bar(
            x + i * width - (len(model_names) - 1) * width / 2,
            values,
            width * 0.9,
            label=model_name.upper(),
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#ccccee",
                fontweight="bold",
            )

    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Model Comparison", fontsize=16, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.legend(facecolor="#1a1a3e", edgecolor="#444477", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return _save_figure(fig, "model_comparison.png", save_dir)
