from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from src.models.train_logistic import (
    build_logistic_pipeline,
    prepare_train_test_split,
    save_model,
)

MODEL_PATH = Path("models/logistic_pipeline.joblib")

FULL_TABLE_PATH = Path("reports/tables/odds_ratio_full.csv")
SUMMARY_TABLE_PATH = Path("reports/tables/odds_ratio_summary.csv")
PLOT_PATH = Path("reports/figures/models/odds_ratio_plot.png")


SELECTED_FEATURES = [
    "num__tenure",
    "cat__Contract_Two year",
    "cat__Contract_Month-to-month",
    "cat__InternetService_Fiber optic",
    "cat__InternetService_DSL",
    "cat__PaymentMethod_Electronic check",
    "num__MonthlyCharges",
    "num__TotalCharges",
]

PRETTY_NAME_MAP = {
    "num__tenure": "Tenure (antigüedad del cliente)",
    "cat__Contract_Two year": "Contrato: Two year",
    "cat__Contract_Month-to-month": "Contrato: Month-to-month",
    "cat__InternetService_Fiber optic": "InternetService: Fiber optic",
    "cat__InternetService_DSL": "InternetService: DSL",
    "cat__PaymentMethod_Electronic check": "PaymentMethod: Electronic check",
    "num__MonthlyCharges": "MonthlyCharges",
    "num__TotalCharges": "TotalCharges",
}

INTERPRETATION_MAP = {
    "num__tenure": (
        "Una mayor antigüedad del cliente se asocia con una reducción muy marcada "
        "de los odds de churn. La interpretación corresponde a un incremento de una "
        "desviación estándar, ya que la variable fue escalada."
    ),
    "cat__Contract_Two year": (
        "Tener un contrato de dos años se asocia con una reducción clara del riesgo "
        "estimado de churn."
    ),
    "cat__Contract_Month-to-month": (
        "El contrato mensual se asocia con un aumento importante del riesgo estimado de churn."
    ),
    "cat__InternetService_Fiber optic": (
        "La categoría Fiber optic se asocia con mayores odds de churn dentro del modelo multivariable."
    ),
    "cat__InternetService_DSL": (
        "La categoría DSL se asocia con menor riesgo estimado de churn dentro del modelo."
    ),
    "cat__PaymentMethod_Electronic check": (
        "Electronic check se asocia con un aumento moderado del riesgo estimado de churn."
    ),
    "num__MonthlyCharges": (
        "El efecto de MonthlyCharges cambia respecto al análisis bivariante, lo que sugiere "
        "interacción o redundancia con otras variables del modelo."
    ),
    "num__TotalCharges": (
        "TotalCharges mantiene una asociación positiva, pero su interpretación debe hacerse "
        "con cautela por su relación estructural con tenure y MonthlyCharges."
    ),
}


def load_or_train_logistic_model(model_path: str | Path = MODEL_PATH):
    """
    Load trained logistic pipeline if it exists; otherwise train and save it.
    """
    model_path = Path(model_path)

    if model_path.exists():
        return joblib.load(model_path)

    X_train, X_test, y_train, y_test = prepare_train_test_split()
    clf = build_logistic_pipeline()
    clf.fit(X_train, y_train)
    save_model(clf, model_path)

    return clf


def build_odds_ratio_table(model) -> pd.DataFrame:
    """
    Extract coefficients and odds ratios from the trained logistic pipeline.
    """
    preprocessor = model.named_steps["preprocessor"]
    logistic_model = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = logistic_model.coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
    })

    coef_df["odds_ratio"] = np.exp(coef_df["coefficient"])
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df["direction"] = coef_df["odds_ratio"].apply(
        lambda x: "Aumenta riesgo de churn" if x > 1 else "Reduce riesgo de churn"
    )

    coef_df = coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    return coef_df


def build_selected_summary(full_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a presentation-ready summary of selected variables.
    """
    summary = full_df[full_df["feature"].isin(SELECTED_FEATURES)].copy()

    summary["variable"] = summary["feature"].map(PRETTY_NAME_MAP)
    summary["interpretation"] = summary["feature"].map(INTERPRETATION_MAP)

    summary["coefficient"] = summary["coefficient"].round(3)
    summary["odds_ratio"] = summary["odds_ratio"].round(3)

    presentation_order = [
        "Tenure (antigüedad del cliente)",
        "Contrato: Two year",
        "Contrato: Month-to-month",
        "InternetService: Fiber optic",
        "InternetService: DSL",
        "PaymentMethod: Electronic check",
        "MonthlyCharges",
        "TotalCharges",
    ]

    summary["order"] = summary["variable"].apply(lambda x: presentation_order.index(x))
    summary = summary.sort_values("order").drop(columns=["order", "abs_coefficient"])

    return summary[
        ["variable", "feature", "coefficient", "odds_ratio", "direction", "interpretation"]
    ].reset_index(drop=True)


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def plot_odds_ratios(summary_df: pd.DataFrame, path: str | Path = PLOT_PATH) -> None:
    """
    Save a horizontal bar plot of selected odds ratios.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = summary_df.copy()
    plot_df["color"] = plot_df["direction"].map({
        "Aumenta riesgo de churn": "#d95f02",
        "Reduce riesgo de churn": "#1b9e77",
    })

    plot_df = plot_df.sort_values("odds_ratio", ascending=True)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(
        plot_df["variable"],
        plot_df["odds_ratio"],
        color=plot_df["color"],
    )

    plt.axvline(x=1, linestyle="--", linewidth=1.5)

    for bar, value in zip(bars, plot_df["odds_ratio"]):
        plt.text(
            value + 0.03,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=10,
        )

    plt.title("Odds Ratios de variables clave - Regresión logística")
    plt.xlabel("Odds Ratio")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    model = load_or_train_logistic_model()

    full_df = build_odds_ratio_table(model)
    summary_df = build_selected_summary(full_df)

    save_table(full_df, FULL_TABLE_PATH)
    save_table(summary_df, SUMMARY_TABLE_PATH)
    plot_odds_ratios(summary_df, PLOT_PATH)

    print("=== Global explainability completed ===")
    print(f"Saved full table to: {FULL_TABLE_PATH}")
    print(f"Saved summary table to: {SUMMARY_TABLE_PATH}")
    print(f"Saved odds ratio plot to: {PLOT_PATH}")
    print("\nSelected summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()