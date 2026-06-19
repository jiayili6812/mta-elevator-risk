"""Shared probability and threshold evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calibration_diagnostics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 5,
) -> dict[str, object]:
    """Return fixed-width development-only reliability diagnostics."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)
    reliability = []
    expected_calibration_error = 0.0
    for index in range(bins):
        mask = assignments == index
        count = int(mask.sum())
        if count:
            mean_probability = float(probabilities[mask].mean())
            observed_rate = float(y_true[mask].mean())
            expected_calibration_error += (
                count / len(y_true) * abs(mean_probability - observed_rate)
            )
        else:
            mean_probability = None
            observed_rate = None
        reliability.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "rows": count,
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
            }
        )
    return {
        "bins": reliability,
        "expected_calibration_error": float(expected_calibration_error),
        "mean_probability": float(probabilities.mean()),
        "observed_prevalence": float(y_true.mean()),
    }


def evaluate_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "threshold": float(threshold),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
