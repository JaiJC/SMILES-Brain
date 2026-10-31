"""Does the affinity model predict *which receptors* a held-out molecule binds?

This evaluates Stage 1 on its own terms, with no brain map and no CBF readout involved. For each
held-out molecule with at least three measured receptors, compare the predicted and measured
affinity *profiles* across those receptors:

* within-molecule Spearman rho between predicted and measured pChEMBL (profile shape), and
* top-1 hit: is the predicted strongest receptor the measured strongest one (chance = 1/k).

The decisive control is the **receptor-mean baseline**: predict every molecule with the training
mean of each receptor. Some receptors are simply bound more tightly by everything in ChEMBL, so a
profile ranking can look good without knowing anything about the molecule. The model has to beat
that, and uncertainty is a cluster bootstrap over Bemis-Murcko scaffolds because analogues share
a scaffold and are not independent samples.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a), rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra @ rb) / denom) if denom > 1e-12 else float("nan")


def spearman_rows(pred: np.ndarray, true: np.ndarray, min_measured: int = 3) -> np.ndarray:
    """Per-row Spearman rho over the receptors measured for that row (NaN if too few/constant).

    `pred` may be (n, k) or a single (k,) vector shared by every row (a baseline).
    """
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if pred.ndim == 1:
        pred = np.broadcast_to(pred, true.shape)
    if pred.shape != true.shape:
        raise ValueError(f"pred {pred.shape} and true {true.shape} must match")
    out = np.full(len(true), np.nan)
    for i in range(len(true)):
        m = np.isfinite(true[i]) & np.isfinite(pred[i])
        if m.sum() >= min_measured:
            out[i] = _rank_corr(pred[i][m], true[i][m])
    return out


def top1_rows(pred: np.ndarray, true: np.ndarray, min_measured: int = 3
              ) -> tuple[np.ndarray, np.ndarray]:
    """Per-row (hit, chance): predicted-argmax == measured-argmax among measured receptors.

    Rows with ties at the measured maximum count a hit if the prediction picks any tied receptor
    (chance is adjusted to tied/k). Rows with fewer than `min_measured` receptors are NaN.
    """
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if pred.ndim == 1:
        pred = np.broadcast_to(pred, true.shape)
    hit = np.full(len(true), np.nan)
    chance = np.full(len(true), np.nan)
    for i in range(len(true)):
        m = np.isfinite(true[i]) & np.isfinite(pred[i])
        k = int(m.sum())
        if k < min_measured:
            continue
        t, p = true[i][m], pred[i][m]
        winners = np.flatnonzero(t == t.max())
        hit[i] = float(int(np.argmax(p)) in winners)
        chance[i] = len(winners) / k
    return hit, chance


def cluster_bootstrap_mean(values: np.ndarray, clusters: np.ndarray, n_boot: int = 2000,
                           seed: int = 0) -> tuple[float, float, float]:
    """Mean of `values` (NaN ignored) with a 95% CI from resampling whole clusters."""
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    ok = np.isfinite(values)
    values, clusters = values[ok], clusters[ok]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    uniq, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(uniq))
    counts = np.bincount(inverse, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return float(values.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def paired_cluster_bootstrap(a: np.ndarray, b: np.ndarray, clusters: np.ndarray,
                             n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean paired difference a - b over rows where both are finite, with a cluster-bootstrap CI."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return cluster_bootstrap_mean(a - b, clusters, n_boot, seed)


def knn_profile_predict(similarity: np.ndarray, train_labels: np.ndarray, k: int = 25) -> np.ndarray:
    """Nearest-analogue baseline: per receptor, the similarity-weighted mean of the k most similar
    training molecules *that have that receptor measured*.

    `similarity` is (n_test, n_train) Tanimoto; `train_labels` is (n_train, n_receptors) with NaN
    where unmeasured. A receptor with no measured training molecule is NaN. This is the simplest
    "copy what close analogues do" model; a learned model has to beat it to justify itself.
    """
    similarity = np.asarray(similarity, dtype=float)
    labels = np.asarray(train_labels, dtype=float)
    if similarity.shape[1] != labels.shape[0]:
        raise ValueError("similarity columns must match the number of training molecules")
    out = np.full((similarity.shape[0], labels.shape[1]), np.nan)
    for r in range(labels.shape[1]):
        cols = np.flatnonzero(np.isfinite(labels[:, r]))
        if cols.size == 0:
            continue
        sim = similarity[:, cols]
        kk = min(k, cols.size)
        top = np.argpartition(-sim, kk - 1, axis=1)[:, :kk]
        w = np.take_along_axis(sim, top, axis=1)
        vals = labels[cols[top], r]
        total = w.sum(axis=1)
        out[:, r] = np.where(total > 1e-12, (w * vals).sum(axis=1) / np.maximum(total, 1e-12),
                             vals.mean(axis=1))
    return out
