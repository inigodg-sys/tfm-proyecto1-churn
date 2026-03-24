from __future__ import annotations

from pathlib import Path
import pandas as pd

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.models.train_logistic import build_logistic_pipeline, prepare_train_test_split


OUTPUT_PATH = Path("reports/tables/threshold_analysis.csv")


def evaluate_thresholds(
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """
    Train logistic regression and evaluate multiple classification thresholds.

    Parameters
    ----------
    thresholds : list[float] | None
        List of thresholds to evaluate. If None, default thresholds are used.

    Returns
    -------
    pd.DataFrame
        Threshold comparison table.
    """
    if thresholds is None:
        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    X_train, X_test, y_train, y_test = prepare_train_test_split()

    clf = build_logistic_pipeline()
    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_test)[:, 1]

    rows = []

    for thr in thresholds:
        y_pred_thr = (y_proba >= thr).astype(int)

        rows.append({
            "threshold": thr,
            "accuracy": accuracy_score(y_test, y_pred_thr),
            "precision": precision_score(y_test, y_pred_thr, zero_division=0),
            "recall": recall_score(y_test, y_pred_thr, zero_division=0),
            "f1": f1_score(y_test, y_pred_thr, zero_division=0),
        })

    threshold_df = pd.DataFrame(rows)
    threshold_df[["accuracy", "precision", "recall", "f1"]] = threshold_df[
        ["accuracy", "precision", "recall", "f1"]
    ].round(4)

    return threshold_df


def save_threshold_table(df: pd.DataFrame, path: str | Path = OUTPUT_PATH) -> None:
    """
    Save threshold analysis table to CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    threshold_df = evaluate_thresholds()
    save_threshold_table(threshold_df)

    print("=== Threshold analysis completed ===")
    print(threshold_df.to_string(index=False))


if __name__ == "__main__":
    main()