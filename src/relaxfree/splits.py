"""Train / calibration / test partitions for the three evaluation regimes."""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

Split = tuple[np.ndarray, np.ndarray, np.ndarray]


def random_splits(n: int, n_repeats: int, seed: int, frac_train: float = 0.5, frac_cal: float = 0.25) -> Iterator[Split]:
    """Defects are exchangeable: every host contributes to all three parts."""
    rng = np.random.default_rng(seed)
    n_train, n_cal = int(frac_train * n), int(frac_cal * n)
    for _ in range(n_repeats):
        perm = rng.permutation(n)
        yield perm[:n_train], perm[n_train:n_train + n_cal], perm[n_train + n_cal:]


def _host_partition(hosts: np.ndarray, test_hosts: set, rng: np.random.Generator, frac_cal: float) -> Split:
    remaining = np.array(sorted(set(hosts) - test_hosts))
    rng.shuffle(remaining)
    n_cal = max(1, int(round(frac_cal * remaining.size)))
    cal_hosts = set(remaining[:n_cal])
    test = np.flatnonzero(np.isin(hosts, list(test_hosts)))
    cal = np.flatnonzero(np.isin(hosts, list(cal_hosts)))
    train = np.flatnonzero(~np.isin(hosts, list(test_hosts | cal_hosts)))
    return train, cal, test


def leave_host_out_splits(df: pd.DataFrame, n_repeats: int, seed: int, frac_cal: float = 0.25) -> Iterator[Split]:
    """Test on one unseen host; calibrate on hosts that the model has not seen either."""
    rng = np.random.default_rng(seed)
    hosts = df["host"].to_numpy()
    for _ in range(n_repeats):
        for h in sorted(set(hosts)):
            yield _host_partition(hosts, {h}, rng, frac_cal)


def leave_family_out_splits(df: pd.DataFrame, n_repeats: int, seed: int, frac_cal: float = 0.25) -> Iterator[Split]:
    """Test on a whole chemical family; calibration hosts come from the other two."""
    rng = np.random.default_rng(seed)
    hosts = df["host"].to_numpy()
    for _ in range(n_repeats):
        for fam in sorted(df["family"].unique()):
            yield _host_partition(hosts, set(df.loc[df["family"] == fam, "host"]), rng, frac_cal)


def leave_element_out_splits(df: pd.DataFrame, n_repeats: int, seed: int, frac_cal: float = 0.25) -> Iterator[Split]:
    """Test on one unseen impurity species; calibrate on species the model has not seen either.

    Native defects always stay in the training set, so calibration and test
    both consist of impurity defects.
    """
    rng = np.random.default_rng(seed)
    impurity = df["defect_class"].isin(["sub_impurity", "int_impurity"]).to_numpy()
    element = df["element"].to_numpy()
    species = np.array(sorted(set(element[impurity])))
    for _ in range(n_repeats):
        for x in species:
            remaining = species[species != x].copy()
            rng.shuffle(remaining)
            cal_species = remaining[: max(1, int(round(frac_cal * remaining.size)))]
            test = np.flatnonzero(impurity & (element == x))
            cal = np.flatnonzero(impurity & np.isin(element, cal_species))
            train = np.setdiff1d(np.arange(len(df)), np.concatenate([test, cal]))
            yield train, cal, test


def make_splits(regime: str, df: pd.DataFrame, n_repeats: int, seed: int) -> Iterator[Split]:
    if regime == "random":
        return random_splits(len(df), n_repeats, seed)
    if regime == "host":
        return leave_host_out_splits(df, n_repeats, seed)
    if regime == "family":
        return leave_family_out_splits(df, n_repeats, seed)
    if regime == "element":
        return leave_element_out_splits(df, n_repeats, seed)
    raise ValueError(f"unknown regime {regime!r}")
