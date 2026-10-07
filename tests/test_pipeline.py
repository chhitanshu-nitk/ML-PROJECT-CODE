import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ids.experiment import filter_classes  # noqa: E402
from ids.evaluate import compute_metrics, per_class_metrics  # noqa: E402
from ids.features import FeaturePipeline, balanced_indices  # noqa: E402
from ids.utils import load_config  # noqa: E402


def test_balanced_indices_equalises_classes():
    rng = np.random.default_rng(0)
    y = np.array([0] * 100 + [1] * 10 + [2] * 3)
    sel = balanced_indices(y, None, rng)
    assert np.bincount(y[sel]).tolist() == [100, 100, 100]
    sel = balanced_indices(y, 20, rng)
    assert np.bincount(y[sel]).tolist() == [20, 20, 20]


def test_pipeline_shapes():
    cfg = load_config()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 20)).astype(np.float32)
    y = np.array([0] * 400 + [1] * 150 + [2] * 50)
    X[y == 1] += 2
    pipe = FeaturePipeline(cfg, "proposal", 0)
    Z, yz = pipe.fit_resample(X, y, 300)
    assert Z.shape == (900, 10) and np.bincount(yz).tolist() == [300, 300, 300]
    assert pipe.transform(X[:50]).shape == (50, 10)
    base = FeaturePipeline(cfg, "baseline", 0)
    Zb, yb = base.fit_resample(X, y)
    assert Zb.shape == X.shape and np.array_equal(yb, y)


def test_custom_class_filter_relabels_remaining_classes():
    X = np.arange(20).reshape(5, 4)
    y = np.array([0, 1, 2, 3, 1])
    filtered_X, filtered_y, names, excluded = filter_classes(
        X, y, ["BENIGN", "RareA", "RareB", "Attack"], ["RareB"]
    )
    assert filtered_X.shape == (4, 4)
    assert filtered_y.tolist() == [0, 1, 2, 1]
    assert names == ["BENIGN", "RareA", "Attack"]
    assert excluded == ["RareB"]


def test_metrics_include_balanced_and_per_class_results():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    proba = np.eye(3, dtype=np.float32)[y_pred]
    metrics = compute_metrics(y_true, y_pred, proba, 3)
    assert {"balanced_accuracy", "mcc"}.issubset(metrics)
    class_metrics = per_class_metrics(y_true, y_pred, ["a", "b", "c"])
    assert list(class_metrics["class"]) == ["a", "b", "c"]
    assert "recall_class_accuracy" in class_metrics
