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


def test_merged_blocks_coarsen_until_calibratable():
    from relaxfree.conformal import REST, assign_blocks, merged_blocks

    cells = np.array(["a|x"] * 10 + ["b|x"] * 5 + ["c|x"] * 5 + ["a|y"] * 3 + ["b|z"] * 2)
    parents = np.array([c.split("|")[1] for c in cells])
    hits = np.ones(cells.size, dtype=bool)
    block = merged_blocks(cells, parents, hits, n_min=9)
    assert block["a|x"] == "a|x"                       # enough hits on its own
    assert block["b|x"] == block["c|x"] == "x|merged"   # 5 + 5 hits when merged within the parent
    assert block["a|y"] == block["b|z"] == REST         # parents y and z stay short
    labels = assign_blocks(block, np.array(["a|x", "d|x", "d|y", "d|w"]), np.array(["x", "x", "y", "w"]))
    assert labels.tolist() == ["a|x", "x|merged", REST, REST]


def test_merged_blocks_ignore_non_hits():
    from relaxfree.conformal import REST, merged_blocks

    cells = np.array(["a|x"] * 20)
    hits = np.array([True] * 8 + [False] * 12)
    assert merged_blocks(cells, np.array(["x"] * 20), hits, n_min=9)["a|x"] == REST


def test_merged_margin_is_valid_per_block():
    from relaxfree.conformal import assign_blocks, merged_blocks

    rng = np.random.default_rng(3)
    alpha, tau, misses, hits = 0.1, 0.0, {}, {}
    scale = {"a|x": 1.0, "b|x": 3.0, "c|x": 0.3}
    size = {"a|x": 60, "b|x": 6, "c|x": 6}
    for _ in range(400):
        def draw(mult):
            cells = np.concatenate([[c] * (n * mult) for c, n in size.items()])
            proxy = np.concatenate([rng.exponential(scale[c], n * mult) for c, n in size.items()])
            return cells, proxy
        c_cal, p_cal = draw(1)
        c_test, p_test = draw(5)
        par_cal, par_test = np.array(["x"] * c_cal.size), np.array(["x"] * c_test.size)
        y_cal, y_test = np.full(c_cal.size, -1.0), np.full(c_test.size, -1.0)   # every candidate is a hit
        block = merged_blocks(c_cal, par_cal, y_cal <= tau, min_calibration_hits(alpha))
        b_cal, b_test = assign_blocks(block, c_cal, par_cal), assign_blocks(block, c_test, par_test)
        margin = grouped_recall_margin(p_cal, y_cal, b_cal, tau, alpha)
        keep = keep_mask(p_test, margin, tau, b_test)
        for b in np.unique(b_test):
            misses[b] = misses.get(b, 0) + int((~keep[b_test == b]).sum())
            hits[b] = hits.get(b, 0) + int((b_test == b).sum())
    assert set(hits) == {"a|x", "x|merged"}
    for b in hits:
        assert misses[b] / hits[b] <= alpha + 0.01
