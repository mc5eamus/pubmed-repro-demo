from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


FRAMINGHAM_COLUMNS = [
    "male",
    "age",
    "education",
    "currentSmoker",
    "cigsPerDay",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
    "TenYearCHD",
]
CLEVELAND_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]


def load_framingham(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(path, na_values=["NA", "?"])
    frame = frame.dropna().drop_duplicates().reset_index(drop=True)
    if "TenYearCHD" not in frame or len(frame.columns) != len(FRAMINGHAM_COLUMNS):
        raise ValueError("Unexpected Framingham columns; expected 15 features and TenYearCHD.")
    features = frame.drop(columns="TenYearCHD")
    target = frame["TenYearCHD"].astype(int)
    return features, target


def load_cleveland(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(
        path,
        header=None,
        names=CLEVELAND_COLUMNS,
        na_values=["?"],
        skipinitialspace=True,
    )
    frame = frame.dropna().reset_index(drop=True)
    features = frame.drop(columns="num").astype(float)
    target = (frame["num"].astype(float) > 0).astype(int)
    return features, target


def stratified_split(features, target, *, test_size: float, seed: int):
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=seed,
        stratify=target,
    )
