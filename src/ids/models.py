from __future__ import annotations

from typing import Any

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

MODEL_NAMES = ["DT", "RF", "ET", "XGB"]


def build_model(name: str, cfg: dict, seed: int, n_classes: int):
    over = dict((cfg.get("model_params") or {}).get(name, {}) or {})
    n_jobs = cfg.get("n_jobs", -1)
    if name == "DT":
        params: dict[str, Any] = dict(random_state=seed); params.update(over)
        return DecisionTreeClassifier(**params)
    if name == "RF":
        params: dict[str, Any] = dict(n_estimators=100, n_jobs=n_jobs, random_state=seed); params.update(over)
        return RandomForestClassifier(**params)
    if name == "ET":
        params: dict[str, Any] = dict(n_estimators=100, n_jobs=n_jobs, random_state=seed); params.update(over)
        return ExtraTreesClassifier(**params)
    if name == "XGB":
        from xgboost import XGBClassifier  # imported lazily so the rest works without it
        params: dict[str, Any] = dict(n_estimators=100, max_depth=6, learning_rate=0.3, tree_method="hist",
                                      n_jobs=n_jobs, random_state=seed,
                                      eval_metric="logloss" if n_classes == 2 else "mlogloss")
        params.update(over)
        return XGBClassifier(**params)
    raise ValueError(f"Unknown model {name}")
