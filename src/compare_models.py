from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

# Reutilizamos funciones del baseline (no ejecuta main por el guard)
from baseline import load_and_clean, make_splits, recall_at_top_k_percent


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUT_PATH = ROOT / "reports" / "model_metrics_valid.csv"


def build_preprocessor_dense(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # Dense para compatibilidad con modelos de árbol/boosting:
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def eval_model(name: str, pipe: Pipeline, X_train, y_train, X_valid, y_valid) -> dict:
    t0 = time.time()
    pipe.fit(X_train, y_train)
    fit_s = time.time() - t0

    y_proba = pipe.predict_proba(X_valid)[:, 1]
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_valid, y_proba),
        "pr_auc": average_precision_score(y_valid, y_proba),
        "recall_at_top10": recall_at_top_k_percent(y_valid, y_proba, k=0.10),
        "fit_seconds": fit_s,
    }


def main() -> None:
    X, y = load_and_clean(DATA_PATH)
    split = make_splits(X, y, seed=42)

    pre = build_preprocessor_dense(split.X_train)

    models = {
        "logreg": LogisticRegression(max_iter=3000),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "hist_gb": HistGradientBoostingClassifier(
            random_state=42,
            max_depth=6,
            learning_rate=0.05,
            max_iter=400,
        ),
    }

    rows = []
    for name, clf in models.items():
        pipe = Pipeline([("preprocess", pre), ("model", clf)])
        rows.append(
            eval_model(
                name, pipe,
                split.X_train, split.y_train,
                split.X_valid, split.y_valid
            )
        )

    df = pd.DataFrame(rows).sort_values("recall_at_top10", ascending=False)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print("== Model comparison (VALID) ==")
    print(df.to_string(index=False))
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()