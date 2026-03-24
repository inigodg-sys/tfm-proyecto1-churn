from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

from src.data.load_data import load_telco_data
from src.data.clean_data import clean_telco_data
from src.features.feature_lists import TARGET, CAT_COLS, NUM_COLS
from src.models.metrics import recall_at_top_k
from src.models.baseline import majority_class_baseline, random_topk_baseline


OUTPUT_PATH = Path("reports/tables/model_comparison.csv")


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


def build_logistic_pipeline() -> Pipeline:
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

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])


def build_random_forest_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUM_COLS),
        ("cat", categorical_transformer, CAT_COLS)
    ])

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])


def build_hgb_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUM_COLS),
        ("cat", categorical_transformer, CAT_COLS)
    ])

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])


def evaluate_classifier(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str,
) -> dict:
    """
    Fit model and evaluate on test set using threshold 0.50.
    """
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "Modelo": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_proba),
        "PR_AUC": average_precision_score(y_test, y_proba),
        "Recall@Top10%": recall_at_top_k(y_test, y_proba, k=0.10),
    }


def build_comparison_table() -> pd.DataFrame:
    """
    Train/evaluate all models and return a comparison dataframe.
    """
    X_train, X_test, y_train, y_test = prepare_train_test_split()

    # Baselines
    baseline_majority = majority_class_baseline(y_test)
    baseline_random = random_topk_baseline(y_test, k=0.10, random_state=42)

    rows = []

    rows.append({
        "Modelo": "Baseline mayoritario",
        "Accuracy": baseline_majority["accuracy"],
        "Precision": baseline_majority["precision"],
        "Recall": baseline_majority["recall"],
        "ROC_AUC": np.nan,
        "PR_AUC": np.nan,
        "Recall@Top10%": baseline_random["recall_at_top_k"],
    })

    # Logistic Regression
    clf_log = build_logistic_pipeline()
    rows.append(evaluate_classifier(
        clf_log, X_train, X_test, y_train, y_test, "Regresión logística"
    ))

    # Random Forest
    clf_rf = build_random_forest_pipeline()
    rows.append(evaluate_classifier(
        clf_rf, X_train, X_test, y_train, y_test, "Random Forest"
    ))

    # HistGradientBoosting
    clf_hgb = build_hgb_pipeline()
    rows.append(evaluate_classifier(
        clf_hgb, X_train, X_test, y_train, y_test, "HistGradientBoosting"
    ))

    comparison_df = pd.DataFrame(rows)

    metric_cols = ["Accuracy", "Precision", "Recall", "ROC_AUC", "PR_AUC", "Recall@Top10%"]
    comparison_df[metric_cols] = comparison_df[metric_cols].round(4)

    return comparison_df


def save_comparison_table(df: pd.DataFrame, path: str | Path = OUTPUT_PATH) -> None:
    """
    Save model comparison table to CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    comparison_df = build_comparison_table()
    save_comparison_table(comparison_df)

    print("=== Model comparison completed ===")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()