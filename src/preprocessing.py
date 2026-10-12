from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "creditcard.csv"


def load_dataset(path=DATA_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("Place creditcard.csv inside the data folder.")
    return pd.read_csv(path)


def drop_duplicate_features(df):
    df = df.drop_duplicates().copy()
    return df.loc[:, ~df.T.duplicated()]


def clean_data(df):
    df = drop_duplicate_features(df)
    df.columns = [str(column).strip() for column in df.columns]
    return df


def feature_engineering(df):
    df = df.copy()
    if "Amount" in df.columns:
        df["Amount_Log"] = np.log1p(df["Amount"])
    if "Time" in df.columns:
        df["Hour"] = (df["Time"] // 3600) % 24
    return df


def split_features_target(df):
    df = clean_data(df)
    df = feature_engineering(df)

    if "Class" not in df.columns:
        raise ValueError("Target column 'Class' was not found.")

    x = df.drop(columns=["Class"])
    y = df["Class"].astype(int)
    return x, y