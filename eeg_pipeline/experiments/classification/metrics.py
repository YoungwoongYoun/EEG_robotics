"""Dependency-light classification metrics and confusion matrices."""

from __future__ import annotations

import numpy as np


def confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if targets.shape != predictions.shape or targets.ndim != 1:
        raise ValueError("targets and predictions must be matching 1D arrays")
    if np.any((targets < 0) | (targets >= n_classes)):
        raise ValueError("targets contain an invalid class")
    if np.any((predictions < 0) | (predictions >= n_classes)):
        raise ValueError("predictions contain an invalid class")
    encoded = targets * n_classes + predictions
    return np.bincount(encoded, minlength=n_classes**2).reshape(n_classes, n_classes)


def classification_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    n_classes: int,
) -> tuple[dict[str, float], np.ndarray]:
    """Compute accuracy, macro-F1, and Cohen's kappa."""

    matrix = confusion_matrix(targets, predictions, n_classes)
    total = int(matrix.sum())
    if total == 0:
        raise ValueError("Cannot evaluate an empty prediction set")

    true_positive = np.diag(matrix).astype(np.float64)
    predicted_count = matrix.sum(axis=0).astype(np.float64)
    target_count = matrix.sum(axis=1).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros_like(true_positive),
        where=predicted_count != 0,
    )
    recall = np.divide(
        true_positive,
        target_count,
        out=np.zeros_like(true_positive),
        where=target_count != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) != 0,
    )
    accuracy = float(true_positive.sum() / total)
    expected_agreement = float((target_count @ predicted_count) / (total**2))
    kappa = (
        float((accuracy - expected_agreement) / (1 - expected_agreement))
        if expected_agreement < 1.0
        else 0.0
    )
    metrics = {
        "accuracy": accuracy,
        "macro_f1": float(f1.mean()),
        "cohen_kappa": kappa,
    }
    for class_id in range(n_classes):
        metrics[f"class_{class_id}_recall"] = float(recall[class_id])
    return metrics, matrix
