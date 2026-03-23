from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score


# ---------- Config ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


@dataclass(frozen=True)
class SplitData:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray


def load_and_clean(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Input : CSV original
    Output: X (features) + y (target binario 0/1)
    """
    df = pd.read_csv(path)

    # 1) Target -> binario
    y = (df["Churn"] == "Yes").astype(int).to_numpy()

    # 2) Eliminamos columnas no predictivas / target leakage trivial
    df = df.drop(columns=["Churn", "customerID"])

    # 3) TotalCharges suele venir como texto con espacios ('' o ' ')
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")

    # 4) Filas con tenure = 0 suelen tener TotalCharges vacío -> 0 es coherente
    mask_tenure0 = df["tenure"] == 0
    df.loc[mask_tenure0, "TotalCharges"] = df.loc[mask_tenure0, "TotalCharges"].fillna(0.0)

    return df, y


def make_splits(X: pd.DataFrame, y: np.ndarray, seed: int = 42) -> SplitData:
    """
    Split estratificado: mantiene el % de churn en train/valid/test.
    """
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp
    )
    return SplitData(X_train, X_valid, X_test, y_train, y_valid, y_test)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Define transformaciones separadas para numéricas y categóricas.
    """
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
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )
    return preprocessor


def recall_at_top_k_percent(y_true: np.ndarray, y_score: np.ndarray, k: float = 0.10) -> float:
    """
    Recall@TopK%: si solo actúo sobre el K% con mayor score,
    ¿qué fracción de churners reales capturo?
    """
    if not (0 < k <= 1):
        raise ValueError("k debe estar en (0, 1].")

    n = len(y_true)
    top_n = int(np.ceil(k * n))

    idx_sorted = np.argsort(y_score)[::-1]  # scores desc
    idx_top = idx_sorted[:top_n]

    tp_top = y_true[idx_top].sum()
    positives = y_true.sum()

    return float(tp_top / positives) if positives > 0 else 0.0


def main() -> None:
    # 1) Load + clean
    X, y = load_and_clean(DATA_PATH)

    # 2) Split
    split = make_splits(X, y, seed=42)

    # 3) Preprocess + model (baseline)
    preprocessor = build_preprocessor(split.X_train)

    clf = LogisticRegression(max_iter=2000, n_jobs=None)

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", clf),
        ]
    )

    # 4) Train
    model.fit(split.X_train, split.y_train)

    # 5) Evaluate on VALID (para decidir; test es para el final)
    y_proba_valid = model.predict_proba(split.X_valid)[:, 1]

    roc = roc_auc_score(split.y_valid, y_proba_valid)
    pr = average_precision_score(split.y_valid, y_proba_valid)
    r10 = recall_at_top_k_percent(split.y_valid, y_proba_valid, k=0.10)

    print("== Baseline: Logistic Regression (VALID) ==")
    print(f"ROC-AUC        : {roc:.4f}")
    print(f"PR-AUC         : {pr:.4f}")
    print(f"Recall@Top10%  : {r10:.4f}")


if __name__ == "__main__":
    main()