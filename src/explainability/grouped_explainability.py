from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from src.models.train_logistic import (
    build_logistic_pipeline,
    prepare_train_test_split,
    save_model,
)

MODEL_PATH = Path("models/logistic_pipeline.joblib")

TOP10_PROFILE_PATH = Path("reports/tables/top10_group_profile.csv")
TOP10_PROTECTIVE_PATH = Path("reports/tables/top10_group_protective.csv")


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


def build_grouped_explainability_tables(
    k: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build grouped explainability tables for the top-k highest-risk customers
    versus the rest of the test set.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        top10_profile_df: variables that characterize the top-k risk group
        top10_protective_df: variables that most separate the rest from top-k
    """
    X_train, X_test, y_train, y_test = prepare_train_test_split()
    model = load_or_train_logistic_model()

    preprocessor = model.named_steps["preprocessor"]
    logistic_model = model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    coefs = logistic_model.coef_

    X_test_transformed = preprocessor.transform(X_test)
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()

    X_test_transformed_df = pd.DataFrame(
        X_test_transformed,
        columns=feature_names,
        index=X_test.index,
    )

    # Local contributions for each observation
    contrib_df = X_test_transformed_df.mul(coefs, axis=1)

    # Probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    local_summary = pd.DataFrame(
        {"y_true": y_test, "y_proba": y_proba},
        index=X_test.index,
    )

    # Top-k group
    n_top = max(1, int(len(local_summary) * k))
    top_idx = local_summary.sort_values("y_proba", ascending=False).head(n_top).index
    rest_idx = local_summary.drop(index=top_idx).index

    mean_contrib_top = contrib_df.loc[top_idx].mean()
    mean_contrib_rest = contrib_df.loc[rest_idx].mean()

    group_compare = pd.DataFrame({
        "feature": feature_names,
        "mean_contrib_top10": mean_contrib_top.values,
        "mean_contrib_rest": mean_contrib_rest.values,
    })

    group_compare["diff_top10_vs_rest"] = (
        group_compare["mean_contrib_top10"] - group_compare["mean_contrib_rest"]
    )

    # Direction tag
    group_compare["group_effect"] = group_compare["diff_top10_vs_rest"].apply(
        lambda x: "Caracteriza Top10 riesgo" if x > 0 else "Aleja del Top10 riesgo"
    )

    # Top variables that characterize risk group
    top10_profile_df = (
        group_compare.sort_values("diff_top10_vs_rest", ascending=False)
        .head(12)
        .reset_index(drop=True)
    )

    # Variables that most separate away from top10 risk
    top10_protective_df = (
        group_compare.sort_values("diff_top10_vs_rest", ascending=True)
        .head(12)
        .reset_index(drop=True)
    )

    # Add metadata columns
    for df_out in (top10_profile_df, top10_protective_df):
        df_out["topk_fraction"] = k
        df_out["topk_size"] = n_top
        df_out["min_probability_in_topk"] = round(local_summary.loc[top_idx, "y_proba"].min(), 4)

        numeric_cols = ["mean_contrib_top10", "mean_contrib_rest", "diff_top10_vs_rest"]
        df_out[numeric_cols] = df_out[numeric_cols].round(6)

    return top10_profile_df, top10_protective_df


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    top10_profile_df, top10_protective_df = build_grouped_explainability_tables(k=0.10)

    save_table(top10_profile_df, TOP10_PROFILE_PATH)
    save_table(top10_protective_df, TOP10_PROTECTIVE_PATH)

    print("=== Grouped explainability completed ===")
    print(f"Saved top10 risk profile table to: {TOP10_PROFILE_PATH}")
    print(f"Saved protective/opposite table to: {TOP10_PROTECTIVE_PATH}")

    print("\nTop variables que más caracterizan al Top10 de riesgo:")
    print(top10_profile_df.to_string(index=False))

    print("\nVariables que más alejan del Top10 de riesgo:")
    print(top10_protective_df.to_string(index=False))


if __name__ == "__main__":
    main()