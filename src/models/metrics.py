import numpy as np
import pandas as pd


def recall_at_top_k(
    y_true: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
    k: float = 0.10,
) -> float:
    """
    Compute Recall@TopK%.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray
        Ground truth binary labels (0/1).
    scores : pd.Series | np.ndarray
        Model scores or probabilities for the positive class.
    k : float
        Fraction of highest-risk observations to keep (e.g. 0.10 = top 10%).

    Returns
    -------
    float
        Recall captured inside the top-k fraction.

    Raises
    ------
    ValueError
        If k is not in (0, 1].
    """
    if not (0 < k <= 1):
        raise ValueError("k must be in the interval (0, 1].")

    y_true_arr = np.asarray(y_true)
    scores_arr = np.asarray(scores)

    n_top = max(1, int(len(y_true_arr) * k))
    top_idx = np.argsort(scores_arr)[::-1][:n_top]

    captured_positives = y_true_arr[top_idx].sum()
    total_positives = y_true_arr.sum()

    if total_positives == 0:
        return 0.0

    return float(captured_positives / total_positives)