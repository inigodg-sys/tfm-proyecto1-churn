from pathlib import Path
import pandas as pd


DEFAULT_PROCESSED_DATA_PATH = Path("data/processed/telco_clean.csv")


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the Telco Customer Churn dataset.

    Cleaning steps
    --------------
    1. Drop `customerID`
    2. Convert `TotalCharges` to numeric
    3. Detect implicit missing values in `TotalCharges`
    4. If `tenure == 0` and `TotalCharges` is missing, impute with 0
    5. If any remaining missing values exist in `TotalCharges`, impute with median

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """
    df_clean = df.copy()

    # 1) Drop identifier if present
    if "customerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["customerID"])

    # 2) Convert TotalCharges to numeric if present
    if "TotalCharges" not in df_clean.columns:
        raise KeyError("Column 'TotalCharges' not found in dataframe.")

    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")

    # 3) Impute implicit missing values linked to tenure == 0
    mask_tenure_zero = (df_clean["tenure"] == 0) & (df_clean["TotalCharges"].isna())
    df_clean.loc[mask_tenure_zero, "TotalCharges"] = 0

    # 4) Safety fallback: if anything remains missing, use median
    if df_clean["TotalCharges"].isna().sum() > 0:
        median_total_charges = df_clean["TotalCharges"].median()
        df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(median_total_charges)

    return df_clean


def save_clean_data(
    df: pd.DataFrame,
    path: str | Path = DEFAULT_PROCESSED_DATA_PATH,
    index: bool = False,
) -> None:
    """
    Save cleaned dataset to disk.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.
    path : str | Path
        Output path.
    index : bool
        Whether to save index.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)