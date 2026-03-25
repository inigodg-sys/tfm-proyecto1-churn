from pathlib import Path
import pandas as pd


# Raíz del proyecto: subimos desde src/data/load_data.py hasta la carpeta principal
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


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
        raw_dir = PROJECT_ROOT / "data" / "raw"
        available_files = list(raw_dir.glob("*")) if raw_dir.exists() else []
        available_names = [f.name for f in available_files]

        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            f"Available files in data/raw: {available_names}"
        )

    return pd.read_csv(path)