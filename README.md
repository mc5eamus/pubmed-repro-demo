# pubmed-repro-demo

This repository reproduces the leakage-controlled coronary heart disease (CHD)
analysis in Raman S., Thakkar D., et al. (2026), *Machine Learning for Coronary
Heart Disease Prediction: Comparative Analysis of Framingham and Cleveland
Subset of the UCI Dataset with SHAP-Based Interpretability*
([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC13298444/), DOI
[10.3390/epidemiologia7030075](https://doi.org/10.3390/epidemiologia7030075)).

## Run the analysis

Requires Python 3.10 or later. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m raman2026_chd
```

The command downloads the raw datasets into `data/raw/`, verifies their
SHA-256 checksums, and records the source URLs and hashes in
`data/raw/sources.json`. Raw data is excluded from version control. The full
run performs five-fold model tuning for the single holdout and 100 fixed-
hyperparameter stratified splits per dataset; it writes the comparison and
robustness report to `raman2026_chd/REPORT.md` and plots/tables to
`raman2026_chd/results/`.

Useful options:

```bash
python -m raman2026_chd --splits 10
python -m raman2026_chd --skip-tuning --splits 1 --bootstrap-replicates 100
python -m raman2026_chd --skip-repeats
```

Run the leakage-control and data-cleaning tests with:

```bash
python -m pytest raman2026_chd/tests
```

All analysis choices not specified in the paper—including seeds, the CV
scoring measure and grid ranges, resampling ratios, neural-network details,
threshold, bootstrap method, and repeated-split protocol—are documented in
`raman2026_chd/config.json` and the generated report.