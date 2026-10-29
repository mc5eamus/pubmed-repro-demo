import argparse
import hashlib
import json
from pathlib import Path

from .analysis import (
    evaluate_holdout,
    repeat_splits,
    summarize_repeats,
    write_report,
)
from .data import load_cleveland, load_framingham, stratified_split
from .download_data import RAW_DIR, SOURCES, download_data


PACKAGE_DIR = Path(__file__).resolve().parent
def main():
    parser = argparse.ArgumentParser(description="Reproduce Raman et al. (2026) CHD analyses.")
    parser.add_argument("--splits", type=int, default=None, help="Number of repeated splits (default: config value).")
    parser.add_argument("--skip-tuning", action="store_true", help="Use published fixed hyperparameters for the single holdout.")
    parser.add_argument("--skip-repeats", action="store_true", help="Skip the repeated-split robustness analysis.")
    parser.add_argument("--bootstrap-replicates", type=int, default=None)
    args = parser.parse_args()

    config = json.loads((PACKAGE_DIR / "config.json").read_text())
    if args.bootstrap_replicates is not None:
        config["bootstrap_replicates"] = args.bootstrap_replicates
    split_count = (
        0
        if args.skip_repeats
        else config["splits"] if args.splits is None else args.splits
    )
    if split_count < 1 and not args.skip_repeats:
        parser.error("--splits must be at least one unless --skip-repeats is used")
    required_data = [*SOURCES, "sources.json"]
    if not all((RAW_DIR / name).is_file() for name in required_data):
        download_data()
    else:
        for name, source in SOURCES.items():
            checksum = hashlib.sha256((RAW_DIR / name).read_bytes()).hexdigest()
            if checksum != source["sha256"]:
                raise ValueError(
                    f"Checksum mismatch for {RAW_DIR / name}; rerun "
                    "`python -m raman2026_chd.download_data` to replace it."
                )

    datasets = {
        "framingham": load_framingham(RAW_DIR / "framingham.csv"),
        "cleveland": load_cleveland(RAW_DIR / "processed.cleveland.data"),
    }
    results_dir = PACKAGE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    holdout_summaries = {}
    repeat_summaries = {}
    rank_frequencies = {}
    split_counts = {}
    for offset, (dataset_name, (features, target)) in enumerate(datasets.items()):
        train_x, test_x, train_y, test_y = stratified_split(
            features,
            target,
            test_size=config["test_size"],
            seed=config["seed"],
        )
        split_counts[dataset_name] = {
            "rows": len(features),
            "expected_rows": 3658 if dataset_name == "framingham" else 297,
            "train": len(train_x),
            "expected_train": 2926 if dataset_name == "framingham" else 237,
            "test": len(test_x),
            "expected_test": 732 if dataset_name == "framingham" else 60,
            "train_positive_rate": float(train_y.mean()),
            "test_positive_rate": float(test_y.mean()),
        }
        print(
            f"{dataset_name}: {len(features)} cleaned rows; "
            f"{len(train_x)} train / {len(test_x)} test"
        )
        summary, _ = evaluate_holdout(
            dataset_name,
            train_x,
            train_y,
            test_x,
            test_y,
            config=config,
            output_dir=results_dir,
            seed=config["seed"],
            tune=not args.skip_tuning,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        holdout_summaries[dataset_name] = summary
        if not args.skip_repeats:
            repeated = repeat_splits(
                dataset_name,
                features,
                target,
                config=config,
                splits=split_count,
                seed=config["seed"] + 1000 * offset,
            )
            repeated.to_csv(results_dir / f"{dataset_name}_repeated_splits.csv", index=False)
            summarized, ranks = summarize_repeats(repeated)
            summarized.to_csv(results_dir / f"{dataset_name}_repeat_summary.csv", index=False)
            repeat_summaries[dataset_name] = summarized
            rank_frequencies[dataset_name] = ranks
    (results_dir / "holdout_metrics.json").write_text(
        json.dumps(holdout_summaries, indent=2, default=lambda value: value.item()) + "\n"
    )
    write_report(
        PACKAGE_DIR / "REPORT.md",
        config,
        holdout_summaries,
        repeat_summaries,
        rank_frequencies,
        split_counts,
        split_count,
    )
    print(f"Report written to {PACKAGE_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
