"""Step 1 of the paper: cleaning / preprocessing of the raw CIC-IDS2017 CSV.

* strip spaces from column names
* coerce everything to numeric, drop rows with NaN / +-inf (paper: "eradicating rows containing null, -inf, inf")
* drop duplicate rows (keep first)
* merge similar low-count classes (the three Web Attack variants -> "Web Attack")
* shrink memory: float64 -> float32
Output: X.npy (float32), y.npy (int16 class ids), meta.json  -> consumed by the experiments.
Works chunk-wise so the full ~2.8M-row file never needs to be parsed in one go.
"""
from __future__ import annotations

import json
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .utils import get_logger, resolve, save_json

log = get_logger()


def clean_column_name(col) -> str:
    name = "_".join(str(col).strip().split())
    return name.replace("/", "_per_") or "unnamed"


def clean_columns(columns: Iterable) -> List[str]:
    used: Dict[str, int] = {}
    out = []
    for c in columns:
        n = clean_column_name(c)
        k = used.get(n, 0)
        used[n] = k + 1
        out.append(n if k == 0 else f"{n}_{k}")
    return out


def normalize_label(s, merge_web: bool = True) -> str:
    s = str(s).strip()
    low = s.lower()
    if merge_web and low.startswith("web attack"):
        return "Web Attack"
    if low == "benign":
        return "BENIGN"
    return s


def preprocess(cfg: dict, nrows: int | None = None) -> dict:
    d = cfg["data"]
    raw = resolve(d["raw_csv"])
    if not raw.exists():
        raise FileNotFoundError(f"Raw CSV not found: {raw}\nPut the file there or edit data.raw_csv in config.yaml")
    out_dir = resolve(d["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    target = clean_column_name(d["label_column"])
    merge_web = bool(d.get("merge_web_attacks", True))

    stats = {"rows_read": 0, "rows_dropped_nan_inf_or_bad_label": 0}
    feats = None
    frames = []
    reader = pd.read_csv(raw, chunksize=int(d["chunk_size"]), low_memory=False,
                         skipinitialspace=True, nrows=nrows)
    for i, chunk in enumerate(reader):
        chunk.columns = clean_columns(chunk.columns)
        if target not in chunk.columns:
            raise ValueError(f"Label column '{target}' not found. Columns end with: {list(chunk.columns)[-5:]}")
        stats["rows_read"] += len(chunk)
        y = chunk[target].astype(str).map(lambda s: normalize_label(s, merge_web))
        X = (chunk.drop(columns=[target])
                  .apply(pd.to_numeric, errors="coerce")
                  .replace([np.inf, -np.inf], np.nan))
        bad_label = y.str.upper().isin({target.upper(), "NAN", ""})
        keep = (~X.isna().any(axis=1)) & (~bad_label)
        stats["rows_dropped_nan_inf_or_bad_label"] += int((~keep).sum())
        X = X.loc[keep].astype(np.float32)
        if feats is None:
            feats = list(X.columns)
        else:
            X = X[feats]
        X["__label__"] = y.loc[keep].values
        frames.append(X)
        log.info("chunk %d processed (%d rows read so far)", i, stats["rows_read"])

    df = pd.concat(frames, ignore_index=True)
    del frames
    df["__label__"] = df["__label__"].astype("category")
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    stats["duplicate_rows_removed"] = n_before - len(df)

    y = df.pop("__label__").astype(str)
    mc = int(d.get("min_class_count", 0))
    if mc > 0:
        counts = y.value_counts()
        rare = counts[counts < mc].index
        y = y.where(~y.isin(rare), "Other")
        stats["merged_into_Other"] = list(rare)

    others = sorted(c for c in y.unique() if c != "BENIGN")
    classes = (["BENIGN"] if (y == "BENIGN").any() else []) + others
    mapping = {c: i for i, c in enumerate(classes)}
    y_int = y.map(mapping).to_numpy(np.int16)
    X = df.to_numpy(dtype=np.float32)

    np.save(out_dir / "X.npy", X)
    np.save(out_dir / "y.npy", y_int)
    meta = {
        "source_csv": str(raw),
        "feature_names": feats,
        "classes": classes,
        "benign_index": mapping.get("BENIGN"),
        "n_rows": int(len(y_int)),
        "n_features": int(X.shape[1]),
        "class_counts": {c: int((y_int == i).sum()) for c, i in mapping.items()},
        "cleaning": stats,
    }
    save_json(meta, out_dir / "meta.json")
    log.info("Saved %s rows x %d features, %d classes -> %s", len(y_int), X.shape[1], len(classes), out_dir)
    for c, n in meta["class_counts"].items():
        log.info("  %-18s %9d", c, n)
    return meta


def load_processed(cfg: dict):
    out_dir = resolve(cfg["data"]["processed_dir"])
    if not (out_dir / "X.npy").exists():
        raise FileNotFoundError(f"No processed data in {out_dir}. Run scripts/01_preprocess.py first.")
    X = np.load(out_dir / "X.npy")
    y = np.load(out_dir / "y.npy")
    with open(out_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return X, y, meta
