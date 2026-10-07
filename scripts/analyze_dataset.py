"""Inspect the dataset and explain the configured experiment pipeline.

Run from the project root:
    python scripts/analyze_dataset.py
    python scripts/analyze_dataset.py --output data/processed/dataset_analysis.json
"""
import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ids.data import clean_column_name, clean_columns, normalize_label
from ids.utils import load_config, resolve


def counts(values):
    return {str(k): int(v) for k, v in pd.Series(values).value_counts().items()}


def inspect_raw(cfg, nrows=None):
    d = cfg["data"]
    raw = resolve(d["raw_csv"])
    if not raw.exists():
        return {"status": "missing", "path": str(raw)}

    target = clean_column_name(d["label_column"])
    frames = []
    raw_columns = None
    stats = {
        "rows_read": 0,
        "rows_dropped_for_invalid_feature_or_label": 0,
        "rows_before_duplicate_removal": 0,
        "duplicate_rows_removed": 0,
        "rows_after_cleaning": 0,
    }
    reader = pd.read_csv(raw, chunksize=int(d["chunk_size"]), low_memory=False,
                         skipinitialspace=True, nrows=nrows)
    for chunk in reader:
        stats["rows_read"] += len(chunk)
        if raw_columns is None:
            raw_columns = list(chunk.columns)
        chunk.columns = clean_columns(chunk.columns)
        if target not in chunk.columns:
            raise ValueError(f"Label column '{target}' was not found")
        labels = chunk[target].astype(str).map(
            lambda value: normalize_label(value, bool(d.get("merge_web_attacks", True)))
        )
        features = (chunk.drop(columns=[target]).apply(pd.to_numeric, errors="coerce")
                    .replace([np.inf, -np.inf], np.nan))
        bad_label = labels.str.upper().isin({target.upper(), "NAN", ""})
        keep = (~features.isna().any(axis=1)) & (~bad_label)
        stats["rows_dropped_for_invalid_feature_or_label"] += int((~keep).sum())
        valid = features.loc[keep].astype(np.float32)
        valid["__label__"] = labels.loc[keep].values
        frames.append(valid)

    cleaned = pd.concat(frames, ignore_index=True)
    stats["rows_before_duplicate_removal"] = len(cleaned)
    stats["duplicate_rows_removed"] = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates(ignore_index=True)
    stats["rows_after_cleaning"] = len(cleaned)

    labels_before = cleaned["__label__"].copy()
    rare = []
    minimum = int(d.get("min_class_count", 0))
    if minimum > 0:
        class_counts = labels_before.value_counts()
        rare = [str(c) for c in class_counts[class_counts < minimum].index]
        cleaned["__label__"] = cleaned["__label__"].where(
            ~cleaned["__label__"].isin(rare), "Other")

    feature_names = [c for c in cleaned.columns if c != "__label__"]
    stats.update({
        "class_counts_before_rare_merge": counts(labels_before),
        "class_counts_after_cleaning": counts(cleaned["__label__"]),
        "rare_classes_merged_into_Other": rare,
        "raw_columns_including_label": len(raw_columns or []),
        "raw_feature_columns_excluding_label": len(raw_columns or []) - 1,
        "final_feature_columns": len(feature_names),
        "feature_columns_removed": len(raw_columns or []) - 1 - len(feature_names),
        "feature_names": feature_names,
        "path": str(raw),
        "status": "available",
    })
    return stats


def inspect_processed(cfg):
    directory = resolve(cfg["data"]["processed_dir"])
    x_path, y_path, meta_path = directory / "X.npy", directory / "y.npy", directory / "meta.json"
    if not (x_path.exists() and y_path.exists() and meta_path.exists()):
        return {"status": "missing", "path": str(directory)}
    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {
        "status": "available",
        "shape": list(X.shape),
        "feature_dtype": str(X.dtype),
        "label_dtype": str(y.dtype),
        "classes": meta["classes"],
        "class_counts": {name: int((y == i).sum())
                         for i, name in enumerate(meta["classes"])},
        "feature_names": meta["feature_names"],
        "cleaning_stats": meta.get("cleaning", {}),
        "path": str(directory),
    }


def sampling_report(y, cfg, task, profile_name):
    labels = (y != 0).astype(np.int64) if task == "binary" else y
    profile = cfg["pipelines"][profile_name]
    configured_cap = profile[f"max_per_class_{task}"]
    class_counts = np.bincount(labels)
    target = int(class_counts.max()) if configured_cap is None else min(
        int(class_counts.max()), int(configured_cap)
    )
    return {
        "task": task,
        "pipeline": profile_name,
        "cap": configured_cap,
        "counts_before": counts(labels),
        "target_rows_per_class": target,
        "total_rows_after_proposal_sampling": target * len(class_counts),
        "classes_downsampled_without_replacement": [
            str(i) for i, n in enumerate(class_counts) if n > target],
        "classes_oversampled_with_replacement": [
            str(i) for i, n in enumerate(class_counts) if n < target],
    }


def pipeline_explanation(cfg):
    folds = int(cfg["experiment"]["folds"])
    split = "10-fold StratifiedKFold" if folds > 1 else "stratified 80/20 holdout"
    return {
        "preprocessing_order": [
            "Read raw CSV in chunks.",
            "Clean column names and separate the label.",
            "Convert features to numeric and remove rows with NaN, +/-inf, or invalid labels.",
            "Remove duplicate complete rows.",
            "Merge Web Attack labels and optionally merge rare classes into Other.",
            "Save float32 X.npy, integer y.npy, and meta.json.",
        ],
        "paper_pipeline": [
            "Keep every class.",
            "Fit scaling on the complete processed dataset.",
            "Balance the complete dataset in proposal mode.",
            "Fit SFE (K-Means and GMM), append two cluster IDs, and fit PCA.",
            f"Split the transformed data with {split}; duplicate samples may cross folds.",
        ],
        "custom_pipeline": [
            "Remove the configured excluded classes before splitting.",
            f"Split the filtered processed data first with {split}.",
            "Fit scaling only on each training fold.",
            "Balance only the training fold in proposal mode.",
            "Fit SFE and PCA only on training data.",
            "Transform the untouched test fold and evaluate.",
        ],
        "sampling_note": "There is no separate sampler: below-cap classes are oversampled with replacement and above-cap classes are downsampled without replacement.",
        "baseline_note": "Baseline mode only scales features; proposal mode adds balancing, SFE, and PCA.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--raw-csv")
    parser.add_argument("--processed-dir")
    parser.add_argument("--nrows", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.raw_csv:
        cfg["data"]["raw_csv"] = str(Path(args.raw_csv).resolve())
    if args.processed_dir:
        cfg["data"]["processed_dir"] = str(Path(args.processed_dir).resolve())

    report = {
        "raw_dataset": inspect_raw(cfg, args.nrows),
        "processed_dataset": inspect_processed(cfg),
        "protocol_and_order": pipeline_explanation(cfg),
    }
    processed = report["processed_dataset"]
    if processed["status"] == "available":
        y = np.load(resolve(cfg["data"]["processed_dir"]) / "y.npy")
        meta = json.loads(
            (resolve(cfg["data"]["processed_dir"]) / "meta.json").read_text(encoding="utf-8")
        )
        report["sampling"] = {}
        for profile_name, profile in cfg["pipelines"].items():
            keep = np.array([
                name not in set(profile.get("excluded_classes", []))
                for name in meta["classes"]
            ])
            profile_y = y[keep[y]]
            report["sampling"][profile_name] = {
                task: sampling_report(profile_y, cfg, task, profile_name)
                for task in ("binary", "multiclass")
            }
    else:
        report["sampling"] = "Unavailable until processed y.npy exists."

    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"\nSaved report to {output}")


if __name__ == "__main__":
    main()
