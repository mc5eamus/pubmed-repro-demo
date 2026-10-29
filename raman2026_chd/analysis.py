import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from .download_data import SOURCES
from .models import (
    fit_neural_network,
    make_pipeline,
    neural_network_probabilities,
)


PUBLISHED_METRICS = {
    "framingham": {
        "random_forest": [0.7391, 0.2872, 0.4865, 0.3612, 0.2222, 0.6929, 0.1750],
        "logistic_regression": [0.6803, 0.2764, 0.6847, 0.3938, 0.2698, 0.7234, 0.2138],
        "knn": [0.6489, 0.2448, 0.6306, 0.3526, 0.2079, 0.6741, 0.2271],
        "xgboost": [0.6899, 0.2456, 0.5045, 0.3304, 0.1762, 0.6882, 0.2255],
        "feedforward_nn": [0.7719, 0.2812, 0.3243, 0.3013, 0.1664, 0.6625, 0.2048],
    },
    "cleveland": {
        "random_forest": [0.8333, 0.8462, 0.7857, 0.8148, 0.6652, 0.9275, 0.1122],
        "logistic_regression": [0.8667, 0.8571, 0.8571, 0.8571, 0.7321, 0.9475, 0.2096],
        "knn": [0.8333, 0.8462, 0.7857, 0.8148, 0.6652, 0.9531, 0.0942],
        "xgboost": [0.8333, 0.8750, 0.7500, 0.8077, 0.6683, 0.9364, 0.1087],
        "feedforward_nn": [0.8500, 0.8276, 0.8571, 0.8421, 0.6997, 0.8867, 0.1360],
    },
}
METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "mcc", "auroc", "brier"]
CLASSICAL_MODELS = ["logistic_regression", "random_forest", "knn", "xgboost"]


def calculate_metrics(target, probabilities, threshold: float) -> dict:
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(target, predictions),
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "f1": f1_score(target, predictions, zero_division=0),
        "mcc": matthews_corrcoef(target, predictions),
        "auroc": roc_auc_score(target, probabilities),
        "brier": brier_score_loss(target, probabilities),
    }


def bootstrap_intervals(target, probabilities, *, threshold, replicates, seed):
    target = np.asarray(target)
    probabilities = np.asarray(probabilities)
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(target == label) for label in (0, 1)]
    samples = {metric: [] for metric in METRIC_NAMES}
    for _ in range(replicates):
        indexes = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        result = calculate_metrics(
            target[indexes], probabilities[indexes], threshold=threshold
        )
        for metric, value in result.items():
            samples[metric].append(value)
    return {
        metric: np.quantile(values, [0.025, 0.975]).tolist()
        for metric, values in samples.items()
    }


def tune_model(name, train_x, train_y, *, seed, config):
    cv = StratifiedKFold(
        n_splits=config["cv_folds"], shuffle=True, random_state=seed
    )
    search = GridSearchCV(
        make_pipeline(name, {}, seed, config),
        config["tuning_grids"][name],
        scoring=config["cv_scoring"],
        cv=cv,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(train_x, train_y)
    return search.best_estimator_, search.best_params_, search.best_score_


def _probabilities(estimator, features):
    return estimator.predict_proba(features)[:, 1]


def evaluate_holdout(
    dataset_name,
    train_x,
    train_y,
    test_x,
    test_y,
    *,
    config,
    output_dir,
    seed,
    tune=True,
    bootstrap_replicates=None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    probabilities_by_model = {}
    estimators_by_model = {}
    for name in CLASSICAL_MODELS:
        if tune:
            estimator, params, cv_score = tune_model(
                name, train_x, train_y, seed=seed, config=config
            )
        else:
            params = config["published_hyperparameters"][dataset_name][name]
            estimator = make_pipeline(name, params, seed, config).fit(train_x, train_y)
            cv_score = None
        probabilities = _probabilities(estimator, test_x)
        probabilities_by_model[name] = probabilities
        estimators_by_model[name] = estimator
        results[name] = {
            "metrics": calculate_metrics(test_y, probabilities, config["threshold"]),
            "bootstrap_95_ci": bootstrap_intervals(
                test_y,
                probabilities,
                threshold=config["threshold"],
                replicates=bootstrap_replicates or config["bootstrap_replicates"],
                seed=seed,
            ),
            "params": params,
            "cv_roc_auc": cv_score,
        }

    nn_model, nn_scaler = fit_neural_network(
        train_x, train_y, seed=seed, config=config
    )
    nn_probabilities = neural_network_probabilities(nn_model, nn_scaler, test_x)
    probabilities_by_model["feedforward_nn"] = nn_probabilities
    results["feedforward_nn"] = {
        "metrics": calculate_metrics(test_y, nn_probabilities, config["threshold"]),
        "bootstrap_95_ci": bootstrap_intervals(
            test_y,
            nn_probabilities,
            threshold=config["threshold"],
            replicates=bootstrap_replicates or config["bootstrap_replicates"],
            seed=seed,
        ),
        "params": {"hidden_layers": config["neural_network"]["hidden_layers"]},
        "cv_roc_auc": None,
    }
    results["xgboost"]["shap_top_features"] = make_shap_plots(
        dataset_name,
        estimators_by_model["xgboost"],
        test_x,
        output_dir,
        seed=seed,
    )
    plot_roc_curves(dataset_name, test_y, probabilities_by_model, output_dir)
    return results, probabilities_by_model


def make_shap_plots(dataset_name, estimator, test_x, output_dir, *, seed):
    classifier = estimator.named_steps["classifier"]
    explainer = shap.TreeExplainer(classifier)
    values = explainer.shap_values(estimator.named_steps["scaler"].transform(test_x))
    if isinstance(values, list):
        values = values[1]
    elif values.ndim == 3:
        values = values[:, :, 1]
    shap.summary_plot(
        values,
        test_x,
        feature_names=test_x.columns,
        plot_type="bar",
        show=False,
        max_display=15,
        rng=np.random.default_rng(seed),
    )
    plt.tight_layout()
    plt.savefig(output_dir / f"{dataset_name}_shap_bar.png", dpi=150)
    plt.close()
    shap.summary_plot(
        values,
        test_x,
        feature_names=test_x.columns,
        show=False,
        max_display=15,
        rng=np.random.default_rng(seed),
    )
    plt.tight_layout()
    plt.savefig(output_dir / f"{dataset_name}_shap_beeswarm.png", dpi=150)
    plt.close()
    return (
        pd.Series(np.abs(values).mean(axis=0), index=test_x.columns)
        .sort_values(ascending=False)
        .head(10)
        .index.tolist()
    )


def plot_roc_curves(dataset_name, target, predictions, output_dir):
    plt.figure(figsize=(7, 6))
    for name, probabilities in predictions.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(target, probabilities)
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{name} (AUC {roc_auc_score(target, probabilities):.3f})",
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"{dataset_name.title()} held-out ROC curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{dataset_name}_roc.png", dpi=150)
    plt.close()


def repeat_splits(dataset_name, features, target, *, config, splits, seed):
    rows = []
    for repeat in range(splits):
        split_seed = seed + repeat
        train_x, test_x, train_y, test_y = train_test_split(
            features,
            target,
            test_size=config["test_size"],
            random_state=split_seed,
            stratify=target,
        )
        for name in CLASSICAL_MODELS:
            params = config["published_hyperparameters"][dataset_name][name]
            estimator = make_pipeline(name, params, split_seed, config).fit(train_x, train_y)
            metrics = calculate_metrics(
                test_y, _probabilities(estimator, test_x), config["threshold"]
            )
            rows.append({"split": repeat + 1, "model": name, **metrics})
        network, scaler = fit_neural_network(
            train_x, train_y, seed=split_seed, config=config
        )
        metrics = calculate_metrics(
            test_y,
            neural_network_probabilities(network, scaler, test_x),
            config["threshold"],
        )
        rows.append({"split": repeat + 1, "model": "feedforward_nn", **metrics})
    return pd.DataFrame(rows)


def summarize_repeats(frame):
    summaries = []
    for model, subset in frame.groupby("model"):
        row = {"model": model}
        for metric in METRIC_NAMES:
            values = subset[metric]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_sd"] = values.std(ddof=1)
            row[f"{metric}_95_low"], row[f"{metric}_95_high"] = values.quantile(
                [0.025, 0.975]
            ).tolist()
        summaries.append(row)
    ranks = {}
    for metric, ascending in (
        ("accuracy", False),
        ("precision", False),
        ("recall", False),
        ("auroc", False),
        ("brier", True),
        ("f1", False),
    ):
        winners = frame.loc[
            frame.groupby("split")[metric].transform(
                lambda values: values == (values.min() if ascending else values.max())
            )
        ]
        ranks[metric] = (winners["model"].value_counts() / frame["split"].nunique()).to_dict()
    return pd.DataFrame(summaries), ranks


def write_report(
    report_path,
    config,
    dataset_summaries,
    repeat_summaries,
    rank_frequencies,
    split_counts,
    split_count,
):
    lines = [
        "# Raman et al. (2026) CHD reproduction",
        "",
        "This report reproduces [Raman et al. (2026)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13298444/), DOI [10.3390/epidemiologia7030075](https://doi.org/10.3390/epidemiologia7030075), and is generated by `python -m raman2026_chd`. Raw datasets are fetched from the URLs and verified against SHA-256 values in `raman2026_chd/download_data.py`.",
        "",
        "## Methods, assumptions, and deviations",
        "",
        f"- Random seed: `{config['seed']}`; stratified holdout: `{config['test_size']:.0%}`; tuning: {config['cv_folds']}-fold CV scored by `{config['cv_scoring']}`.",
        f"- Each split's seed is also used for SMOTE, random undersampling, and stochastic classifiers. The repeated-split seeds start at `{config['seed']}` for Framingham and `{config['seed'] + 1000}` for Cleveland.",
        f"- SMOTE uses `{config['smote_sampling_strategy']}` then random undersampling uses `{config['undersampling_strategy']}`; scaling is fitted after training-fold resampling. The held-out test data is only transformed, never resampled.",
        f"- Decision threshold: `{config['threshold']}`. Neural net uses hidden layers {config['neural_network']['hidden_layers']}, dropout {config['neural_network']['dropout']}, Adam, batch size {config['neural_network']['batch_size']}, a maximum of {config['neural_network']['epochs']} epochs, patience {config['neural_network']['early_stopping_patience']}, and {config['neural_network']['num_threads']} CPU thread; early stopping monitors training loss (no validation set).",
        "- Repeated splits use the published fixed hyperparameters (no per-split retuning, to keep the CPU analysis practical); ranking intervals are empirical 2.5th–97.5th percentiles across splits. Bootstrap CIs use "
        + f"{config['bootstrap_replicates']:,} stratified resamples of held-out test patients.",
        "- Published tuning ranges are not reported by the paper. We use the explicit, compact search spaces in `config.json` and ROC AUC as the CV scoring measure.",
        "- The source Framingham CSV has 16 columns including the outcome; analysis checks complete cases and removes duplicates. Cleveland `?` values are removed and disease is `num > 0`.",
        "- Dependency versions are pinned in `requirements.txt`; the PyTorch CPU wheel is used for the laptop workflow.",
        "- The paper reports PyTorch 2.12.0; this reproduction pins the CPU build of PyTorch 2.9.1, a documented library-version deviation.",
        "",
        "## Cleaning and holdout counts",
        "",
        "| Dataset | Clean rows (paper) | Train (paper) | Test (paper) | Train positive rate | Test positive rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset, counts in split_counts.items():
        lines.append(
            f"| {dataset.title()} | {counts['rows']} ({counts['expected_rows']}) | {counts['train']} ({counts['expected_train']}) | {counts['test']} ({counts['expected_test']}) | {counts['train_positive_rate']:.3f} | {counts['test_positive_rate']:.3f} |"
        )
    lines.extend(["", "## Single-split metrics versus paper", ""])
    for dataset, results in dataset_summaries.items():
        lines.extend(
            [
                f"### {dataset.title()}",
                "",
                "| Model | Metric | Reproduced (bootstrap 95% CI) | Published | Delta |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for model, result in results.items():
            for index, metric in enumerate(METRIC_NAMES):
                value = result["metrics"][metric]
                published = PUBLISHED_METRICS[dataset][model][index]
                low, high = result["bootstrap_95_ci"][metric]
                lines.append(
                    f"| {model} | {metric} | {value:.4f} [{low:.4f}, {high:.4f}] | {published:.4f} | {value - published:+.4f} |"
                )
        lines.extend(
            [
                "",
                "Single-split configurations (the classical models use the CV-selected "
                "parameters unless `--skip-tuning` was used):",
                "",
            ]
        )
        for model, result in results.items():
            lines.append(f"- {model}: `{json.dumps(result['params'], sort_keys=True)}`")
        lines.extend(
            [
                "",
                f"![{dataset} ROC curves](results/{dataset}_roc.png)",
                "",
                f"![{dataset} SHAP feature importance](results/{dataset}_shap_bar.png)",
                "",
                f"![{dataset} SHAP beeswarm](results/{dataset}_shap_beeswarm.png)",
                "",
                "Paper's expected XGBoost SHAP features: "
                + (
                    "sysBP, currentSmoker, prevalentHyp. "
                    if dataset == "framingham"
                    else "ca, cp, thal, age. "
                )
                + "Reproduced top features: "
                + ", ".join(results["xgboost"]["shap_top_features"])
                + ".",
                "",
            ]
        )
    lines.extend(
        [
            "## Repeated-split robustness",
            "",
            (
                f"{split_count} stratified 80/20 holdout splits per dataset; the published hyperparameters are fixed for every split."
                if split_count
                else "Repeated-split analysis was skipped for this run."
            ),
            "",
        ]
    )
    for dataset, summary in repeat_summaries.items():
        lines.extend(
            [
                f"### {dataset.title()}",
                "",
                "| Model | Metric | Mean | SD | Empirical 95% interval |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for _, row in summary.iterrows():
            for metric in METRIC_NAMES:
                lines.append(
                    f"| {row['model']} | {metric} | {row[f'{metric}_mean']:.4f} | {row[f'{metric}_sd']:.4f} | [{row[f'{metric}_95_low']:.4f}, {row[f'{metric}_95_high']:.4f}] |"
                )
        lines.extend(["", "Ranked first (ties count for each model):", ""])
        for metric, frequencies in rank_frequencies[dataset].items():
            lines.append(
                f"- {metric}: "
                + ", ".join(
                    f"{model} {count:.1%}" for model, count in sorted(frequencies.items())
                )
            )
        claims = {
            "framingham": {
                "accuracy": "feedforward_nn",
                "auroc": "logistic_regression",
                "brier": "random_forest",
            },
            "cleveland": {
                "accuracy": "logistic_regression",
                "precision": "logistic_regression",
                "recall": "logistic_regression",
                "auroc": "knn",
                "brier": "knn",
                "f1": "logistic_regression",
            },
        }[dataset]
        lines.extend(["", "Published headline ranking checks (majority of splits = stable):", ""])
        for metric, claimed_model in claims.items():
            rate = rank_frequencies[dataset][metric].get(claimed_model, 0.0)
            status = "survives" if rate > 0.5 else "does not consistently survive"
            lines.append(
                f"- {metric}: the paper's {claimed_model} claim {status} "
                f"({rate:.1%} of splits ranked first)."
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "The paper's single-split ranking claims should be interpreted alongside the repeated-split rank frequencies above. Each test split is small (especially Cleveland), so per-split ranks can vary substantially. This reproduction does not establish a clinical prediction model.",
            "",
            "## Dataset source checksums",
            "",
            "| File | Source URL | SHA-256 |",
            "|---|---|---|",
        ]
    )
    for filename, source in SOURCES.items():
        lines.append(
            f"| `{filename}` | {source['url']} | `{source['sha256']}` |"
        )
    lines.extend(
        [
            "",
            "The downloader also writes this verified manifest to `../data/raw/sources.json`.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines))
