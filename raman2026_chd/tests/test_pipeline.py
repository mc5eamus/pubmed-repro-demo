import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from raman2026_chd import models
from raman2026_chd.data import load_cleveland, load_framingham


def test_framingham_complete_cases_and_duplicates(tmp_path):
    rows = [
        [0, 40, 1, 0, 0, 0, 0, 0, 0, 200, 120, 80, 25, 70, 80, 0],
        [0, 40, 1, 0, 0, 0, 0, 0, 0, 200, 120, 80, 25, 70, 80, 0],
        [1, 50, 2, 1, 10, 0, 0, 1, 0, 210, 140, 90, 28, 75, np.nan, 1],
        [1, 60, 1, 0, 0, 0, 0, 0, 0, 220, 130, 85, 27, 72, 90, 1],
    ]
    path = tmp_path / "framingham.csv"
    pd.DataFrame(
        rows,
        columns=[
            "male", "age", "education", "currentSmoker", "cigsPerDay", "BPMeds",
            "prevalentStroke", "prevalentHyp", "diabetes", "totChol", "sysBP",
            "diaBP", "BMI", "heartRate", "glucose", "TenYearCHD",
        ],
    ).to_csv(path, index=False)

    features, target = load_framingham(path)

    assert features.shape == (2, 15)
    assert target.tolist() == [0, 1]


def test_cleveland_drops_unknowns_and_binarizes_target(tmp_path):
    path = tmp_path / "processed.cleveland.data"
    path.write_text(
        "63,1,1,145,233,1,2,150,0,2.3,3,0,6,0\n"
        "67,1,4,160,286,0,2,108,1,1.5,2,3,3,2\n"
        "57,1,4,120,354,0,0,163,1,0.6,1,?,7,0\n"
    )

    features, target = load_cleveland(path)

    assert features.shape == (2, 13)
    assert target.tolist() == [0, 1]


def test_resampling_and_scaling_fit_only_on_training_data(monkeypatch):
    calls = {"smote": 0, "undersample": 0}

    class PassThroughSampler(BaseEstimator):
        def __init__(self, sampling_strategy=1.0, random_state=None):
            self.sampling_strategy = sampling_strategy
            self.random_state = random_state

        def fit_resample(self, features, target):
            calls[self.key] += 1
            return features, target

    class SmoteSpy(PassThroughSampler):
        key = "smote"

        def __init__(self, sampling_strategy=1.0, k_neighbors=5, random_state=None):
            super().__init__(sampling_strategy, random_state)
            self.k_neighbors = k_neighbors

    class UndersampleSpy(PassThroughSampler):
        key = "undersample"

    monkeypatch.setattr(models, "SMOTE", SmoteSpy)
    monkeypatch.setattr(models, "RandomUnderSampler", UndersampleSpy)
    config = {
        "smote_sampling_strategy": 1.0,
        "undersampling_strategy": 1.0,
        "smote_neighbors": 5,
    }
    train_x = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    train_y = np.array([0, 0, 1, 1])
    test_x = np.array([[1000.0, 2000.0], [2000.0, 3000.0]])

    pipeline = models.make_pipeline(
        "logistic_regression",
        {"C": 1.0, "penalty": "l2", "solver": "liblinear"},
        seed=3,
        config=config,
    )
    pipeline.fit(train_x, train_y)
    scaler = pipeline.named_steps["scaler"]
    pipeline.predict_proba(test_x)

    np.testing.assert_allclose(scaler.mean_, train_x.mean(axis=0))
    assert calls == {"smote": 1, "undersample": 1}
