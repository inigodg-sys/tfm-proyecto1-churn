from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from src.models.metrics import recall_at_top_k


def majority_class_baseline(y_true: pd.Series | np.ndarray) -> dict:
    """
    Compute baseline metrics for a majority-class classifier
    that always predicts 0 (no churn).

    Parameters
    ----------
    y_true : pd.Series | np.ndarray
        Ground truth binary labels.

    Returns
    -------
    dict
        Dictionary with baseline metrics and confusion matrix.
    """
    y_true_arr = np.asarray(y_true)
    y_pred = np.zeros_like(y_true_arr)

    return {
        "model": "baseline_majority",
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred),
    }


def random_topk_baseline(
    y_true: pd.Series | np.ndarray,
    k: float = 0.10,
    random_state: int = 42,
) -> dict:
    """
    Compute a random baseline for Recall@TopK%.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray
        Ground truth binary labels.
    k : float
        Fraction of top observations to keep.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    dict
        Dictionary with Recall@TopK% result.
    """
    rng = np.random.default_rng(random_state)
    random_scores = rng.random(len(y_true))

    return {
        "model": "baseline_random_topk",
        "k": k,
        "recall_at_top_k": float(recall_at_top_k(y_true, random_scores, k=k)),
    }