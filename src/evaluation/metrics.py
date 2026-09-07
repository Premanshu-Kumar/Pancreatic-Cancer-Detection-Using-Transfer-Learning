"""
Performance Metrics
====================
Computes accuracy, precision, recall, F1-score, confusion matrix,
and ROC-AUC for model evaluation on the test set.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    classification_report,
)

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import config


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute all classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_pred_proba : np.ndarray
        Predicted probabilities (continuous, 0–1).
    threshold : float
        Classification threshold.

    Returns
    -------
    dict
        Dictionary with all computed metrics.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Core metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Specificity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # ROC-AUC
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        auc = 0.0

    # ROC curve data
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)

    # Precision-Recall curve data
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(
        y_true, y_pred_proba
    )

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "specificity": float(specificity),
        "auc_roc": float(auc),
        "confusion_matrix": cm,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "roc_curve": {"fpr": fpr, "tpr": tpr, "thresholds": roc_thresholds},
        "pr_curve": {
            "precision": pr_precision,
            "recall": pr_recall,
            "thresholds": pr_thresholds,
        },
        "threshold": threshold,
    }


def generate_classification_report(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    class_names: list = None,
) -> str:
    """
    Generate a formatted classification report string.

    Parameters
    ----------
    y_true : np.ndarray
        True labels.
    y_pred_proba : np.ndarray
        Predicted probabilities.
    threshold : float
        Classification threshold.
    class_names : list, optional
        Names for classes. Defaults to config.CLASS_NAMES.

    Returns
    -------
    str
        Formatted classification report.
    """
    if class_names is None:
        class_names = config.CLASS_NAMES

    y_pred = (y_pred_proba >= threshold).astype(int)
    return classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )


def evaluate_model(
    model,
    test_generator,
    threshold: float = 0.5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a trained model on the test set.

    Parameters
    ----------
    model : tf.keras.Model or BaseCancerModel
        The trained model. Can be a Keras model or our wrapper.
    test_generator : DirectoryIterator
        Test data generator.
    threshold : float
        Classification threshold.
    verbose : bool
        Print results to console.

    Returns
    -------
    dict
        Complete evaluation results.
    """
    # Handle both raw Keras models and our wrapper
    if hasattr(model, "model"):
        keras_model = model.model
        model_name = model.model_name
    else:
        keras_model = model
        model_name = "model"

    # Get predictions
    test_generator.reset()
    y_pred_proba = keras_model.predict(test_generator, verbose=0)
    y_pred_proba = y_pred_proba.flatten()

    # Get true labels
    y_true = test_generator.classes

    # Ensure arrays match in length
    y_true = y_true[: len(y_pred_proba)]

    # Compute metrics
    metrics = compute_metrics(y_true, y_pred_proba, threshold)

    # Generate classification report
    report = generate_classification_report(y_true, y_pred_proba, threshold)

    if verbose:
        print(f"\n{'='*50}")
        print(f"  Evaluation Results: {model_name.upper()}")
        print(f"{'='*50}")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  F1-Score:    {metrics['f1_score']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TN={metrics['tn']}  FP={metrics['fp']}")
        print(f"    FN={metrics['fn']}  TP={metrics['tp']}")
        print(f"\n{report}")

    return {
        "model_name": model_name,
        "metrics": metrics,
        "report": report,
        "y_true": y_true,
        "y_pred_proba": y_pred_proba,
    }
