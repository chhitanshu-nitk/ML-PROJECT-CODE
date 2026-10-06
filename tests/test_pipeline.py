import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
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
