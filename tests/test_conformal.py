import math

import numpy as np
import pytest

from relaxfree.conformal import (
    conformal_quantile, grouped_recall_margin, keep_mask, min_calibration_hits, recall_margin,
)


def test_quantile_matches_order_statistic():
    scores = np.arange(1, 20)  # n = 19
    assert conformal_quantile(scores, 0.1) == 18  # ceil(20 * 0.9) = 18th smallest
    assert conformal_quantile(scores, 0.05) == 19
    assert math.isinf(conformal_quantile(scores, 0.01))
    assert math.isinf(conformal_quantile(np.array([]), 0.1))


def test_min_calibration_hits():
    assert min_calibration_hits(0.1) == 9
    assert min_calibration_hits(0.05) == 19
    assert math.isinf(conformal_quantile(np.zeros(8), 0.1))
    assert math.isfinite(conformal_quantile(np.zeros(9), 0.1))


def test_margin_uses_hits_only():
    proxy = np.array([1.0, 2.0, 3.5, 9.0, 10.0])
    y = np.array([0.5, 1.5, 2.9, 3.5, 8.0])
    # hits at tau = 3 are the first three; with alpha = 0.25 the quantile is their maximum
    assert recall_margin(proxy, y, tau=3.0, alpha=0.25) == pytest.approx(0.5)


def test_unseen_group_keeps_everything():
    keep = keep_mask(np.array([10.0, 1.0]), {"a": 0.0}, tau=3.0, groups=np.array(["b", "a"]))
    assert keep.tolist() == [True, True]
    keep = keep_mask(np.array([10.0, 5.0]), {"a": 0.0}, tau=3.0, groups=np.array(["a", "a"]))
    assert keep.tolist() == [False, False]


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_miss_rate_is_controlled_on_average(alpha):
    """Monte Carlo check of the label-conditional guarantee with a biased, noisy proxy."""
    rng = np.random.default_rng(0)
    tau, misses, hits = 3.0, 0, 0
    for _ in range(400):
        y = rng.gamma(3.0, 1.2, size=300)
        proxy = y + rng.exponential(0.8, size=300) * (1 + (y < 2))
        cal, test = np.arange(150), np.arange(150, 300)
        margin = recall_margin(proxy[cal], y[cal], tau, alpha)
        keep = keep_mask(proxy[test], margin, tau)
        is_hit = y[test] <= tau
        hits += is_hit.sum()
        misses += (is_hit & ~keep).sum()
    rate = misses / hits
    assert rate <= alpha + 0.01
    assert rate >= alpha - 0.04


def test_grouped_margins_differ():
    rng = np.random.default_rng(1)
    y = rng.uniform(0, 3, size=400)
    groups = np.repeat(["easy", "hard"], 200)
    proxy = y + np.where(groups == "easy", 0.1, 2.0) * rng.random(400)
    margins = grouped_recall_margin(proxy, y, groups, tau=3.0, alpha=0.1)
    assert margins["hard"] > margins["easy"]
