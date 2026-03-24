from __future__ import annotations

from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from src.data.load_data import load_telco_data
from src.data.clean_data import clean_telco_data
from src.features.feature_lists import TARGET, CAT_COLS, NUM_COLS
from src.models.metrics import recall_at_top_k


MODEL_OUTPUT_PATH = Path("models/logistic_pipeline.joblib")
METRICS_OUTPUT_PATH = Path("reports/tables/logistic_metrics.csv")


def build_logistic_pipeline() -> Pipeline:
    """
    Build the preprocessing + logistic regression pipeline.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUM_COLS),
        ("cat", categorical_transformer, CAT_COLS)
    ])

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline


def prepare_train_test_split(
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load, clean, and split the dataset.
    """
    df_raw = load_telco_data()
    df = clean_telco_data(df_raw)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.50,
) -> dict:
    """
    Evaluate the trained model on the test set.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "model": "logistic_regression",
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "recall_at_top10": recall_at_top_k(y_test, y_proba, k=0.10),
        "tn": int(confusion_matrix(y_test, y_pred)[0, 0]),
        "fp": int(confusion_matrix(y_test, y_pred)[0, 1]),
        "fn": int(confusion_matrix(y_test, y_pred)[1, 0]),
        "tp": int(confusion_matrix(y_test, y_pred)[1, 1]),
    }

    return metrics


def save_metrics(metrics: dict, path: str | Path = METRICS_OUTPUT_PATH) -> None:
    """
    Save model metrics as a one-row CSV file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df_metrics = pd.DataFrame([metrics])
    df_metrics.to_csv(path, index=False)


def save_model(model: Pipeline, path: str | Path = MODEL_OUTPUT_PATH) -> None:
    """
    Save trained pipeline to disk.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)


def main() -> None:
    """
    Train, evaluate, and save the logistic regression model.
    """
    X_train, X_test, y_train, y_test = prepare_train_test_split()

    clf = build_logistic_pipeline()
    clf.fit(X_train, y_train)

    metrics = evaluate_model(clf, X_test, y_test, threshold=0.50)

    save_model(clf)
    save_metrics(metrics)

    print("=== Logistic regression training completed ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()