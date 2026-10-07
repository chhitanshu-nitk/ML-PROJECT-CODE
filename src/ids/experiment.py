"""Experiment runner for the paper and the custom strict pipeline.

``paper`` reproduces the paper-style whole-dataset transformation before CV.
``custom`` removes configured classes, splits first, and fits every learned
transform on the training fold only.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

from .data import load_processed
from .evaluate import (
    compute_metrics,
    per_class_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
    predict_with_proba,
)
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


def filter_classes(X, y, class_names, excluded):
    excluded = set(excluded or [])
    if not excluded:
        return X, y, list(class_names), []
    missing = sorted(excluded.difference(class_names))
    if missing:
        raise ValueError(f"Excluded classes are not present in processed metadata: {missing}")
    keep = np.array([name not in excluded for name in class_names], dtype=bool)
    old_to_new = np.full(len(class_names), -1, dtype=np.int64)
    old_to_new[np.flatnonzero(keep)] = np.arange(int(keep.sum()))
    row_keep = keep[y]
    return X[row_keep], old_to_new[y[row_keep]], [
        name for name, include in zip(class_names, keep) if include
    ], sorted(excluded)


def fold_data(protocol, X, y, mode, cap, folds, cfg):
    seed = cfg["random_state"]
    ts = cfg["experiment"].get("holdout_test_size", 0.2)
    if protocol == "paper":
        pipe = FeaturePipeline(cfg, mode, seed)
        Z, yz = pipe.fit_resample(X, y, cap)
        log.info("  transformed dataset: %s (classes: %s)", Z.shape, np.bincount(yz).tolist())
        for k, (tr, te) in enumerate(make_splits(yz, folds, seed, ts)):
            yield k, Z[tr], yz[tr], Z[te], yz[te]
    elif protocol == "custom":
        for k, (tr, te) in enumerate(make_splits(y, folds, seed, ts)):
            pipe = FeaturePipeline(cfg, mode, seed + k)
            Ztr, ytr = pipe.fit_resample(X[tr], y[tr], cap)
            yield k, Ztr, ytr, pipe.transform(X[te]), y[te]
    else:
        raise ValueError(f"Unknown protocol: {protocol}")


def run(cfg, protocol, tasks, modes, models, folds, out_dir, sample_frac=None, max_per_class=None):
    if protocol not in ("paper", "custom"):
        raise ValueError("protocol must be 'paper' or 'custom'")
    out_dir = Path(out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "confusion_matrices").mkdir(parents=True, exist_ok=True)
    seed = cfg["random_state"]
    X, y_multi, meta = load_processed(cfg)
    y_multi = y_multi.astype(np.int64)
    profile = (cfg.get("pipelines") or {}).get(protocol, {})
    classes = list(meta["classes"])
    X, y_multi, classes, excluded_classes = filter_classes(
        X, y_multi, classes, profile.get("excluded_classes", [])
    )
    if sample_frac and sample_frac < 1:
        X, y_multi = subsample(X, y_multi, sample_frac, seed)
    benign_index = classes.index("BENIGN")
    y_bin = (y_multi != benign_index).astype(np.int64)
    log.info("Loaded X=%s, classes=%d, protocol=%s, excluded=%s",
             X.shape, len(classes), protocol, excluded_classes)
    save_json({"protocol": protocol, "tasks": tasks, "modes": modes, "models": models, "folds": folds,
               "max_per_class_override": max_per_class, "sample_frac": sample_frac,
               "excluded_classes": excluded_classes, "pipeline_profile": profile, "config": cfg},
              out_dir / "run_config.json")

    task_defs = {"binary": (y_bin, ["Benign", "Attack"]), "multiclass": (y_multi, classes)}
    rows = []
    class_rows = []
    for task in tasks:
        y, names = task_defs[task]
        n_cls = len(names)
        configured_cap = profile.get(f"max_per_class_{task}")
        cap = max_per_class if max_per_class is not None else configured_cap
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
                    class_df = per_class_metrics(yte, pred, names)
                    class_df.insert(0, "fold", k)
                    class_df.insert(0, "model", name)
                    class_df.insert(0, "mode", mode)
                    class_df.insert(0, "task", task)
                    class_rows.extend(class_df.to_dict("records"))
                    last[name] = (yte, pred, proba)
                    log.info("  fold %d %-3s acc=%.4f macroF1=%.4f auc=%.4f (fit %.1fs)",
                             k, name, m["accuracy"], m["f1"], m["auc"], fit_s)
                pd.DataFrame(rows).to_csv(out_dir / "fold_metrics.csv", index=False)  # incremental save
                pd.DataFrame(class_rows).to_csv(out_dir / "class_metrics.csv", index=False)
                for model_name, (model_yte, model_pred, _) in last.items():
                    cm_df = pd.DataFrame(
                        confusion_matrix(model_yte, model_pred, labels=np.arange(n_cls)),
                        index=names,
                        columns=names,
                    )
                    cm_df.to_csv(
                        out_dir / "confusion_matrices" /
                        f"cm_{task}_{mode}_{model_name}_fold{k}.csv"
                    )
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
