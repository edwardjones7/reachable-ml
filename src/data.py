"""Load and clean UCI Bank Marketing (bank-additional-full.csv) and build the splits."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "bank-additional-full.csv"
PDAYS_SENTINEL = 999  # "never previously contacted", not a real 999-day gap


def load(path=DATA_PATH):
    """Return (X, y). X keeps the original row order, which is chronological."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} missing - run `python data/download.py` first")
    df = pd.read_csv(path, sep=";")
    y = (df.pop("y") == "yes").astype(int)

    # pdays: split the sentinel into its own flag, keep the real gaps as a numeric
    # column with the never-contacted rows missing (imputed inside the pipeline).
    never = df["pdays"] == PDAYS_SENTINEL
    df["never_contacted"] = never.astype(int)
    df["pdays_contacted"] = df["pdays"].where(~never, np.nan)
    df = df.drop(columns="pdays")
    return df, y


def random_split(X, y, test_size=0.2, seed=SEED):
    """Stratified 80/20 split. The test set is touched once, at the end."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def temporal_split(X, y, test_size=0.2):
    """First 80% of rows (earlier campaigns) train, last 20% (later campaigns) test."""
    cut = int(len(X) * (1 - test_size))
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
