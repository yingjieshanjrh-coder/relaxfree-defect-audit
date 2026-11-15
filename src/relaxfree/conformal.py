"""Split-conformal margins that control the miss rate of a screening rule.

The screening rule keeps a candidate when ``proxy - margin <= tau``. A true
hit (``y <= tau``) is missed when its proxy value exceeds ``tau + margin``.
Taking ``tau + margin`` as the conformal quantile of the proxy values of the
calibration *hits* gives a label-conditional guarantee: for a test hit that is
exchangeable with the calibration hits, ``P(kept) >= 1 - alpha``, and the
miss rate is no smaller than ``alpha - 1 / (n_hits + 1)`` when proxy values
are distinct. The margin may be negative if the proxy runs low.
"""
from __future__ import annotations

import math

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample (1 - alpha) quantile; +inf when there are too few scores."""
    scores = np.sort(np.asarray(scores, dtype=float))
    n = scores.size
    k = math.ceil((n + 1) * (1.0 - alpha))
    if n == 0 or k > n:
        return math.inf
    return float(scores[k - 1])


def min_calibration_hits(alpha: float) -> int:
    """Smallest number of calibration hits that yields a finite margin."""
    return math.ceil(1.0 / alpha) - 1


def recall_margin(proxy_cal: np.ndarray, y_cal: np.ndarray, tau: float, alpha: float) -> float:
    """Margin such that at most a fraction alpha of hits is expected to be missed."""
    proxy_cal = np.asarray(proxy_cal, dtype=float)
    y_cal = np.asarray(y_cal, dtype=float)
    hits = y_cal <= tau
    return conformal_quantile(proxy_cal[hits], alpha) - tau


def grouped_recall_margin(
    proxy_cal: np.ndarray, y_cal: np.ndarray, groups_cal: np.ndarray, tau: float, alpha: float
) -> dict:
    """One margin per group (Mondrian calibration)."""
    groups_cal = np.asarray(groups_cal)
    return {
        g: recall_margin(proxy_cal[groups_cal == g], y_cal[groups_cal == g], tau, alpha)
        for g in np.unique(groups_cal)
    }


def keep_mask(proxy: np.ndarray, margin, tau: float, groups: np.ndarray | None = None) -> np.ndarray:
    """Apply the screening rule; unseen groups get an infinite margin (keep everything)."""
    proxy = np.asarray(proxy, dtype=float)
    if isinstance(margin, dict):
        if groups is None:
            raise ValueError("groups are required with per-group margins")
        m = np.array([margin.get(g, math.inf) for g in np.asarray(groups)], dtype=float)
    else:
        m = np.full(proxy.shape, float(margin))
    return (proxy - m <= tau) | np.isinf(m)
