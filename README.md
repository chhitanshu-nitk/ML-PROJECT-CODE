# ML-PROJECT-CODE

## Experiment pipelines

The project has two explicit profiles for the paper **“Machine
learning-based network intrusion detection for big and imbalanced data using
oversampling, stacking feature embedding and feature extraction”**.

### `paper`

This profile is for paper-style reproduction:

- keeps all processed classes;
- fits scaling, balancing, SFE, and PCA on the complete processed dataset;
- performs cross-validation after that transformation;
- uses majority-class random oversampling when proposal mode is selected.

This ordering can allow duplicate oversampled rows to cross folds. It is kept
for reproduction, not as the leak-free estimate.

### `custom`

This is the project’s strict pipeline:

- excludes `Heartbleed` and `Infiltration` before any split;
- splits first using stratification;
- fits scaling, SFE, and PCA on training data only;
- balances training data only;
- leaves the test fold untouched;
- uses caps of 100,000 per binary class and 10,000 per multiclass class.

Both profiles run `baseline` and `proposal` modes. Baseline uses scaled
features; proposal adds class balancing, SFE, and PCA to 10 components.

Each run writes:

- `fold_metrics.csv`: overall metrics for every fold/model;
- `class_metrics.csv`: per-class precision, recall (class accuracy), F1,
  specificity, and support;
- `confusion_matrices/*.csv` and `figures/*.png`: confusion matrices;
- `summary.csv` and `summary_pct.csv`: means and standard deviations across
  folds;
- ROC curves and aggregate reports.

The reported overall metrics include accuracy, balanced accuracy, macro and
weighted F1, macro precision/recall, ROC-AUC, and Matthews correlation
coefficient. `--folds 1` is only a quick holdout smoke test; the configured
default is 10-fold stratified cross-validation.

Run one profile with:

```powershell
python scripts\02_run_experiments.py --protocol custom --name custom_strict
```

Run both after preprocessing with `run_all.sh` from Git Bash.