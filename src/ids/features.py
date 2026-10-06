"""Steps 2-5 of the paper as one fit/transform object.

baseline  ("All Features"): StandardScaler only
proposal                  : StandardScaler -> Random Oversampling -> SFE (KMeans + GMM cluster ids as
                            2 meta-features) -> PCA(10)

Everything heavy is chunked so the full dataset (and the paper-scale ~26M oversampled rows) can be handled.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

CHUNK = 500_000


def balanced_indices(y: np.ndarray, max_per_class, rng: np.random.Generator) -> np.ndarray:
    """Random Oversampling. Every class -> target rows, target = majority size (or min(cap, majority))."""
    classes, counts = np.unique(y, return_counts=True)
    target = int(counts.max())
    if max_per_class:
        target = min(target, int(max_per_class))
    parts = []
    for c, n in zip(classes, counts):
        idx = np.flatnonzero(y == c)
        if n > target:                                   # only when a cap is set
            parts.append(rng.choice(idx, target, replace=False))
        elif n < target:
            parts.append(idx)
            parts.append(rng.choice(idx, target - n, replace=True))
        else:
            parts.append(idx)
    sel = np.concatenate(parts)
    rng.shuffle(sel)
    return sel


class FeaturePipeline:
    def __init__(self, cfg: dict, mode: str = "proposal", seed: int = 42):
        if mode not in ("baseline", "proposal"):
            raise ValueError("mode must be 'baseline' or 'proposal'")
        self.cfg, self.mode, self.seed = cfg, mode, seed
        self.rng = np.random.default_rng(seed)

    # ---------------- scaling ----------------
    def _scale(self, X):
        out = np.empty(X.shape, dtype=np.float32)
        for i in range(0, len(X), CHUNK):
            out[i:i + CHUNK] = self.scaler.transform(X[i:i + CHUNK].astype(np.float64))
        return out

    # ---------------- SFE / PCA ----------------
    def _meta(self, Xs):
        out = np.empty((len(Xs), 2), dtype=np.float32)
        for i in range(0, len(Xs), CHUNK):
            c = Xs[i:i + CHUNK]
            out[i:i + CHUNK, 0] = self.kmeans.predict(c)
            out[i:i + CHUNK, 1] = self.gmm.predict(c)
        return out

    def _project(self, Xs):
        out = np.empty((len(Xs), self.pca.n_components_), dtype=np.float32)
        for i in range(0, len(Xs), CHUNK):
            c = Xs[i:i + CHUNK]
            aug = np.hstack([c, self.meta_scaler.transform(self._meta(c))])
            out[i:i + CHUNK] = self.pca.transform(aug)
        return out

    def _fit_sfe_pca(self, Xr):
        s, p = self.cfg["sfe"], self.cfg["pca"]
        self.kmeans = MiniBatchKMeans(n_clusters=int(s["n_clusters"]), batch_size=int(s["kmeans_batch_size"]),
                                      n_init=3, random_state=self.seed).fit(Xr)
        gi = self.rng.choice(len(Xr), min(int(s["gmm_fit_samples"]), len(Xr)), replace=False)
        self.gmm = GaussianMixture(n_components=int(s["gmm_components"]),
                                   covariance_type=s.get("gmm_covariance", "diag"),
                                   reg_covar=float(s.get("gmm_reg_covar", 1e-3)),
                                   max_iter=100, random_state=self.seed).fit(Xr[gi])
        pi = self.rng.choice(len(Xr), min(int(p["fit_samples"]), len(Xr)), replace=False)
        Xp = Xr[pi]
        meta = self._meta(Xp)
        self.meta_scaler = StandardScaler().fit(meta)
        aug = np.hstack([Xp, self.meta_scaler.transform(meta)])
        k = min(int(p["n_components"]), aug.shape[1], aug.shape[0])
        self.pca = PCA(n_components=k, random_state=self.seed).fit(aug)

    # ---------------- public API ----------------
    def fit_resample(self, X, y, max_per_class=None):
        """Fit on (X, y) and return the transformed (and, in 'proposal' mode, oversampled) training set."""
        self.scaler = StandardScaler()
        for i in range(0, len(X), CHUNK):
            self.scaler.partial_fit(X[i:i + CHUNK].astype(np.float64))
        Xs = self._scale(X)
        if self.mode == "baseline":
            return Xs, y
        idx = balanced_indices(y, max_per_class, self.rng)
        Xr, yr = Xs[idx], y[idx]
        del Xs
        self._fit_sfe_pca(Xr)
        return self._project(Xr), yr

    def transform(self, X):
        """Transform unseen data (never oversampled)."""
        Xs = self._scale(X)
        return Xs if self.mode == "baseline" else self._project(Xs)
