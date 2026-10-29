import random

import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier


def make_classifier(name: str, params: dict, seed: int):
    if name == "logistic_regression":
        return LogisticRegression(random_state=seed, max_iter=2000, **params)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=1, **params)
    if name == "knn":
        return KNeighborsClassifier(n_jobs=1, **params)
    if name == "xgboost":
        return XGBClassifier(
            random_state=seed,
            n_jobs=1,
            eval_metric="logloss",
            tree_method="hist",
            **params,
        )
    raise ValueError(f"Unknown classifier: {name}")


def make_pipeline(name: str, params: dict, seed: int, config: dict) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "smote",
                SMOTE(
                    sampling_strategy=config["smote_sampling_strategy"],
                    k_neighbors=config["smote_neighbors"],
                    random_state=seed,
                ),
            ),
            (
                "undersample",
                RandomUnderSampler(
                    sampling_strategy=config["undersampling_strategy"],
                    random_state=seed,
                ),
            ),
            ("scaler", StandardScaler()),
            ("classifier", make_classifier(name, params, seed)),
        ]
    )


def resample_and_scale(train_x, train_y, *, seed: int, config: dict):
    smote = SMOTE(
        sampling_strategy=config["smote_sampling_strategy"],
        k_neighbors=config["smote_neighbors"],
        random_state=seed,
    )
    under = RandomUnderSampler(
        sampling_strategy=config["undersampling_strategy"], random_state=seed
    )
    x_balanced, y_balanced = smote.fit_resample(train_x, train_y)
    x_balanced, y_balanced = under.fit_resample(x_balanced, y_balanced)
    scaler = StandardScaler().fit(x_balanced)
    return scaler.transform(x_balanced), np.asarray(y_balanced), scaler


class FeedForwardNetwork(nn.Module):
    def __init__(self, feature_count: int, hidden_layers: list[int], dropout: float):
        super().__init__()
        layers = []
        input_size = feature_count
        for width in hidden_layers:
            layers.extend([nn.Linear(input_size, width), nn.ReLU(), nn.Dropout(dropout)])
            input_size = width
        layers.append(nn.Linear(input_size, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, features):
        return self.layers(features).squeeze(1)


def fit_neural_network(train_x, train_y, *, seed: int, config: dict):
    torch.set_num_threads(config["neural_network"]["num_threads"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    x_balanced, y_balanced, scaler = resample_and_scale(
        train_x, train_y, seed=seed, config=config
    )
    model_config = config["neural_network"]
    model = FeedForwardNetwork(
        x_balanced.shape[1], model_config["hidden_layers"], model_config["dropout"]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config["learning_rate"])
    loss_fn = nn.BCEWithLogitsLoss()
    dataset = TensorDataset(
        torch.tensor(x_balanced, dtype=torch.float32),
        torch.tensor(y_balanced, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=model_config["batch_size"],
        shuffle=True,
        generator=generator,
    )

    best_loss = float("inf")
    best_state = None
    stale_epochs = 0
    for _ in range(model_config["epochs"]):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        epoch_loss = float(np.mean(losses))
        if epoch_loss < best_loss - 1e-5:
            best_loss = epoch_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= model_config["early_stopping_patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, scaler


def neural_network_probabilities(model, scaler, features) -> np.ndarray:
    scaled = scaler.transform(features)
    with torch.no_grad():
        logits = model(torch.tensor(scaled, dtype=torch.float32))
        return torch.sigmoid(logits).numpy()
