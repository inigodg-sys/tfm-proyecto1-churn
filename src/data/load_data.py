from pathlib import Path
import pandas as pd


DEFAULT_RAW_DATA_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def load_telco_data(path: str | Path = DEFAULT_RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn dataset.

    Parameters
    ----------
    path : str | Path
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded raw dataset.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)
    return df