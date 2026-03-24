from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.models.train_logistic import build_logistic_pipeline, prepare_train_test_split
from src.models.metrics import recall_at_top_k


OUTPUT_PATH = Path("reports/tables/business_impact_scenarios.csv")


def get_top_k_capture_summary(
    y_true: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
    k: float = 0.10,
) -> dict:
    """
    Compute capture summary inside the top-k fraction.

    Parameters
    ----------
    y_true : pd.Series | np.ndarray
        Ground truth binary labels.
    scores : pd.Series | np.ndarray
        Scores or probabilities for the positive class.
    k : float
        Top-k fraction.

    Returns
    -------
    dict
        Summary with top-k size, total positives, captured positives,
        and recall_at_top_k.
    """
    y_true_arr = np.asarray(y_true)
    scores_arr = np.asarray(scores)

    n_top = max(1, int(len(y_true_arr) * k))
    top_idx = np.argsort(scores_arr)[::-1][:n_top]

    total_positives = int(y_true_arr.sum())
    captured_positives = int(y_true_arr[top_idx].sum())

    recall_topk = (
        captured_positives / total_positives if total_positives > 0 else 0.0
    )

    return {
        "n_top": n_top,
        "total_positives": total_positives,
        "captured_positives": captured_positives,
        "recall_at_top_k": recall_topk,
    }


def build_business_impact_table(
    k: float = 0.10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Train logistic regression, compute top-k capture performance,
    and translate it into business impact scenarios.
    """
    X_train, X_test, y_train, y_test = prepare_train_test_split()

    clf = build_logistic_pipeline()
    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_test)[:, 1]

    # Model top-k summary
    model_summary = get_top_k_capture_summary(y_test, y_proba, k=k)

    # Random baseline top-k summary
    rng = np.random.default_rng(random_state)
    random_scores = rng.random(len(y_test))
    random_summary = get_top_k_capture_summary(y_test, random_scores, k=k)

    scenarios = [
        {"Escenario": "Conservador", "V": 300, "C": 10, "r": 0.20},
        {"Escenario": "Base",        "V": 500, "C": 15, "r": 0.25},
        {"Escenario": "Agresivo",    "V": 800, "C": 20, "r": 0.30},
    ]

    rows = []

    for s in scenarios:
        V = s["V"]
        C = s["C"]
        r = s["r"]

        n_top = model_summary["n_top"]
        tp_model = model_summary["captured_positives"]
        tp_random = random_summary["captured_positives"]

        impact_model = tp_model * r * V - n_top * C
        impact_random = tp_random * r * V - n_top * C
        impact_incremental = impact_model - impact_random

        rows.append({
            "Escenario": s["Escenario"],
            "TopK": k,
            "Clientes_contactados": n_top,
            "Churners_totales_test": model_summary["total_positives"],
            "Recall@TopK_modelo": round(model_summary["recall_at_top_k"], 4),
            "Recall@TopK_azar": round(random_summary["recall_at_top_k"], 4),
            "Churners_capturados_modelo": tp_model,
            "Churners_capturados_azar": tp_random,
            "Valor_por_cliente_retenido_V": V,
            "Coste_por_contacto_C": C,
            "Tasa_exito_campaña_r": r,
            "Impacto_neto_modelo": round(impact_model, 2),
            "Impacto_neto_azar": round(impact_random, 2),
            "Impacto_incremental_modelo": round(impact_incremental, 2),
        })

    return pd.DataFrame(rows)


def save_business_impact_table(
    df: pd.DataFrame,
    path: str | Path = OUTPUT_PATH,
) -> None:
    """
    Save business impact scenarios table to CSV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    impact_df = build_business_impact_table(k=0.10, random_state=42)
    save_business_impact_table(impact_df)

    print("=== Business impact analysis completed ===")
    print(impact_df.to_string(index=False))


if __name__ == "__main__":
    main()