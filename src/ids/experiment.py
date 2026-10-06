"""Experiment runner.

protocol = "paper"  : preprocess -> scale -> RO -> SFE -> PCA on the WHOLE dataset, then k-fold CV
                      (this is how the paper's numbers were produced; oversampled duplicates end up in both
                      train and test folds, which inflates the scores -> they match the paper)
protocol = "strict" : split first; scaler / RO / SFE / PCA are fitted on the training part only and the
                      held-out part is real, untouched data (leak-free, honest numbers)
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

from .data import load_processed
from .evaluate import compute_metrics, plot_confusion_matrix, plot_roc_curves, predict_with_proba
from .features import FeaturePipeline
from .models import build_model
from .report import write_reports
from .utils import get_logger, save_json

log = get_logger()


def subsample(X, y, frac, seed, min_keep=100):
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        n = min(len(idx), max(int(len(idx) * frac), min_keep))
        keep.append(rng.choice(idx, n, replace=False))
    keep = np.sort(np.concatenate(keep))
    return X[keep], y[keep]


def make_splits(y, folds, seed, test_size):
    if folds <= 1:
        tr, te = train_test_split(np.arange(len(y)), test_size=test_size, stratify=y, random_state=seed)
        yield tr, te
    else:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        yield from skf.split(np.zeros(len(y)), y)


def fold_data(protocol, X, y, mode, cap, folds, cfg):
    seed = cfg["random_state"]
    ts = cfg["experiment"].get("holdout_test_size", 0.2)
    if protocol == "paper":
        pipe = FeaturePipeline(cfg, mode, seed)
        Z, yz = pipe.fit_resample(X, y, cap)
        log.info("  transformed dataset: %s (classes: %s)", Z.shape, np.bincount(yz).tolist())
        for k, (tr, te) in enumerate(make_splits(yz, folds, seed, ts)):
            yield k, Z[tr], yz[tr], Z[te], yz[te]
    else:
        for k, (tr, te) in enumerate(make_splits(y, folds, seed, ts)):
            pipe = FeaturePipeline(cfg, mode, seed + k)
            Ztr, ytr = pipe.fit_resample(X[tr], y[tr], cap)
            yield k, Ztr, ytr, pipe.transform(X[te]), y[te]


def run(cfg, protocol, tasks, modes, models, folds, out_dir, sample_frac=None, max_per_class=None):
    out_dir = Path(out_dir); (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    seed = cfg["random_state"]
    X, y_multi, meta = load_processed(cfg)
    y_multi = y_multi.astype(np.int64)
    if sample_frac and sample_frac < 1:
        X, y_multi = subsample(X, y_multi, sample_frac, seed)
    classes = meta["classes"]
    y_bin = (y_multi != meta["benign_index"]).astype(np.int64)
    log.info("Loaded X=%s, classes=%d, protocol=%s", X.shape, len(classes), protocol)
    save_json({"protocol": protocol, "tasks": tasks, "modes": modes, "models": models, "folds": folds,
               "max_per_class_override": max_per_class, "sample_frac": sample_frac, "config": cfg},
              out_dir / "run_config.json")

    task_defs = {"binary": (y_bin, ["Benign", "Attack"]), "multiclass": (y_multi, classes)}
    rows = []
    for task in tasks:
        y, names = task_defs[task]
        n_cls = len(names)
        cap = max_per_class or cfg["oversampling"][f"max_per_class_{task}"]
        for mode in modes:
            log.info("=== task=%s mode=%s ===", task, mode)
            last = {}
            for k, Ztr, ytr, Zte, yte in fold_data(protocol, X, y, mode, cap, folds, cfg):
                for name in models:
                    try:
                        model = build_model(name, cfg, seed, n_cls)
                    except ImportError as e:
                        log.warning("skip %s (%s)", name, e); continue
                    t0 = time.time(); model.fit(Ztr, ytr); fit_s = time.time() - t0
                    t0 = time.time(); pred, proba = predict_with_proba(model, Zte, n_cls); pred_s = time.time() - t0
                    m = compute_metrics(yte, pred, proba, n_cls)
                    rows.append({"task": task, "mode": mode, "model": name, "fold": k, "n_train": len(ytr),
                                 "n_test": len(yte), **m, "fit_s": fit_s, "pred_s": pred_s})
                    last[name] = (yte, pred, proba)
                    log.info("  fold %d %-3s acc=%.4f macroF1=%.4f auc=%.4f (fit %.1fs)",
                             k, name, m["accuracy"], m["f1"], m["auc"], fit_s)
                pd.DataFrame(rows).to_csv(out_dir / "fold_metrics.csv", index=False)  # incremental save
            for name, (yt, yp, _) in last.items():            # figures from the last fold (as in the paper)
                cm = confusion_matrix(yt, yp, labels=np.arange(n_cls))
                plot_confusion_matrix(cm, names, out_dir / "figures" / f"cm_{task}_{mode}_{name}.png",
                                      f"{task} / {mode} / {name}")
            if last:
                plot_roc_curves({n: (v[0], v[2]) for n, v in last.items()}, n_cls,
                                out_dir / "figures" / f"roc_{task}_{mode}.png", f"ROC {task} / {mode}")
    df = pd.DataFrame(rows)
    comp = write_reports(df, out_dir, protocol)
    log.info("\n%s", comp.to_string(index=False))
    log.info("All outputs written to %s", out_dir)
    return df
