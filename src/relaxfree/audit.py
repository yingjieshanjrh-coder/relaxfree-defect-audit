"""Repeated calibrate-and-screen experiments."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .conformal import (
    assign_blocks, grouped_recall_margin, keep_mask, merged_blocks, min_calibration_hits, recall_margin,
)
from .proxies import PROXIES, fit_predict, proxy_cost
from .screening import wilson_interval
from .splits import make_splits

# calibration name -> column whose values define the calibration groups
CALIBRATION_GROUPS = {"classwise": "defect_class", "cellwise": "cell"}
CALIBRATIONS = ["none", "pooled", *CALIBRATION_GROUPS, "merged"]


@dataclass
class AuditConfig:
    regimes: list[str] = field(default_factory=lambda: ["random", "host", "family"])
    proxies: list[str] = field(default_factory=lambda: list(PROXIES))
    taus: list[float] = field(default_factory=lambda: [2.0, 3.0, 4.0])
    alphas: list[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])
    n_repeats: dict = field(default_factory=lambda: {"random": 50, "host": 5, "family": 10, "element": 5})
    model: str = "hgb"
    seed: int = 7
    calibrations: list[str] = field(default_factory=lambda: list(CALIBRATIONS))
    # smallest number of hits for a block of the merged taxonomy, in units of the minimum
    merge_factor: float = 1.0


def _splits_with_repeat(regime: str, df: pd.DataFrame, n_repeats: int, seed: int):
    impurity = df["defect_class"].isin(["sub_impurity", "int_impurity"])
    per_repeat = {
        "random": 1, "host": df["host"].nunique(), "family": df["family"].nunique(),
        "element": df.loc[impurity, "element"].nunique(),
    }[regime]
    for i, split in enumerate(make_splits(regime, df, n_repeats, seed)):
        yield i // per_repeat, split


def run_audit(df: pd.DataFrame, feats: pd.DataFrame, cfg: AuditConfig) -> pd.DataFrame:
    """Return one row per (configuration, test candidate) with the screening decision."""
    y = df["e_relaxed"].to_numpy()
    cell = (df["family"] + "|" + df["defect_class"]).to_numpy()
    groups = {"defect_class": df["defect_class"].to_numpy(), "cell": cell}
    chunks = []
    for regime in cfg.regimes:
        for rep, (train, cal, test) in _splits_with_repeat(regime, df, cfg.n_repeats[regime], cfg.seed):
            for proxy in cfg.proxies:
                both = np.concatenate([cal, test])
                pred = fit_predict(proxy, df, feats, train, both, cfg.model, cfg.seed + rep)
                p_cal, p_test = pred[: cal.size], pred[cal.size:]
                for tau in cfg.taus:
                    for alpha in cfg.alphas:
                        for calib in cfg.calibrations:
                            level = np.full(test.size, calib, dtype=object)
                            if calib == "none":
                                if alpha != cfg.alphas[0]:
                                    continue
                                keep = p_test <= tau
                                finite = np.ones(test.size, dtype=bool)
                            elif calib == "pooled":
                                m = recall_margin(p_cal, y[cal], tau, alpha)
                                keep = keep_mask(p_test, m, tau)
                                finite = np.full(test.size, np.isfinite(m))
                            elif calib == "merged":
                                n_min = int(np.ceil(cfg.merge_factor * min_calibration_hits(alpha)))
                                cls = groups["defect_class"]
                                block = merged_blocks(cell[cal], cls[cal], y[cal] <= tau, n_min)
                                b_cal = assign_blocks(block, cell[cal], cls[cal])
                                b_test = assign_blocks(block, cell[test], cls[test])
                                m = grouped_recall_margin(p_cal, y[cal], b_cal, tau, alpha)
                                keep = keep_mask(p_test, m, tau, b_test)
                                finite = np.array([np.isfinite(m.get(k, np.inf)) for k in b_test])
                                level = np.where(b_test == cell[test], "cell",
                                                 np.where(b_test == "rest", "rest", "class"))
                            else:
                                g = groups[CALIBRATION_GROUPS[calib]]
                                m = grouped_recall_margin(p_cal, y[cal], g[cal], tau, alpha)
                                keep = keep_mask(p_test, m, tau, g[test])
                                finite = np.array([np.isfinite(m.get(k, np.inf)) for k in g[test]])
                            chunks.append(pd.DataFrame({
                                "regime": regime, "rep": rep, "proxy": proxy, "tau": tau,
                                "alpha": np.nan if calib == "none" else alpha, "calibration": calib,
                                "row": test, "keep": keep, "calibrated": finite, "level": level,
                                "abs_err": np.abs(p_test - y[test]),
                            }))
    out = pd.concat(chunks, ignore_index=True)
    for col in ("regime", "proxy", "calibration", "level"):
        out[col] = out[col].astype("category")
    return out


def summarise(decisions: pd.DataFrame, df: pd.DataFrame, by: list[str], single_point_steps: float = 1.0) -> pd.DataFrame:
    """Aggregate decisions over repeats into miss rate, discard fraction and effort saved.

    ``single_point_steps`` is the cost of the ideal-site calculation in units of
    one ionic step of the recorded relaxation.
    """
    cfg_cols = ["regime", "proxy", "tau", "alpha", "calibration"]
    d = decisions.copy()
    rows = d["row"].to_numpy()
    y = df["e_relaxed"].to_numpy()[rows]
    steps = df["n_frames"].to_numpy()[rows].astype(float)
    cost = {p: proxy_cost(p) for p in d["proxy"].unique()}
    sp = d["proxy"].map(cost).astype(float).to_numpy()
    sp = np.where(sp > 0, sp - 1.0 + single_point_steps, 0.0)
    for col in by:
        d[col] = df[col].to_numpy()[rows]
    d["hit"] = y <= d["tau"].to_numpy()
    d["missed"] = d["hit"] & ~d["keep"]
    d["discard"] = ~d["keep"]
    d["steps"] = steps
    d["steps_saved"] = np.where(d["discard"], np.clip(steps - sp, 0.0, None), 0.0)
    d["false_keep"] = d["keep"] & ~d["hit"]
    g = d.groupby(cfg_cols + by, observed=True, dropna=False)
    out = g.agg(
        n=("hit", "size"), n_hit=("hit", "sum"), n_missed=("missed", "sum"), n_discard=("discard", "sum"),
        n_keep=("keep", "sum"), n_false_keep=("false_keep", "sum"), n_calibrated=("calibrated", "sum"),
        steps=("steps", "sum"), steps_saved=("steps_saved", "sum"), mae=("abs_err", "mean"),
    ).reset_index()
    out["miss_rate"] = out["n_missed"] / out["n_hit"].where(out["n_hit"] > 0)
    out["discard_frac"] = out["n_discard"] / out["n"]
    out["steps_saved_frac"] = out["steps_saved"] / out["steps"]
    out["precision"] = 1.0 - out["n_false_keep"] / out["n_keep"].where(out["n_keep"] > 0)
    out["calibrated_frac"] = out["n_calibrated"] / out["n"]
    return out.drop(columns=["steps", "steps_saved", "n_false_keep", "n_calibrated"])


def per_repeat_miss(decisions: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Overall miss rate of each repeat, to check the guarantee on average."""
    cfg_cols = ["regime", "proxy", "tau", "alpha", "calibration", "rep"]
    d = decisions[decisions["calibration"] != "none"].copy()
    y = df["e_relaxed"].to_numpy()[d["row"].to_numpy()]
    d["hit"] = y <= d["tau"].to_numpy()
    d["missed"] = d["hit"] & ~d["keep"]
    g = d.groupby(cfg_cols, observed=True).agg(n_hit=("hit", "sum"), n_missed=("missed", "sum")).reset_index()
    g["miss_rate"] = g["n_missed"] / g["n_hit"].where(g["n_hit"] > 0)
    return g


def add_wilson(table: pd.DataFrame) -> pd.DataFrame:
    lo, hi = zip(*[wilson_interval(int(k), int(n)) for k, n in zip(table["n_missed"], table["n_hit"])])
    return table.assign(miss_lo=lo, miss_hi=hi)


def bootstrap_miss(
    decisions: pd.DataFrame, df: pd.DataFrame, by: list[str], n_boot: int = 2000, seed: int = 0,
) -> pd.DataFrame:
    """Percentile intervals of the miss rate from resampling defects.

    Repeats reuse the same defects, so pooled counts are not independent
    trials. Each defect contributes its number of test appearances as a hit
    and as a missed hit; defects are resampled within each group.
    """
    cfg_cols = ["regime", "proxy", "tau", "alpha", "calibration"]
    d = decisions.copy()
    rows = d["row"].to_numpy()
    d["hit"] = df["e_relaxed"].to_numpy()[rows] <= d["tau"].to_numpy()
    d["missed"] = d["hit"] & ~d["keep"]
    for col in by:
        d[col] = df[col].to_numpy()[rows]
    per_defect = d.groupby(cfg_cols + by + ["row"], observed=True, dropna=False).agg(
        h=("hit", "sum"), m=("missed", "sum")).reset_index()
    rng = np.random.default_rng(seed)
    out = []
    for key, g in per_defect.groupby(cfg_cols + by, observed=True, dropna=False):
        h, m = g["h"].to_numpy(float), g["m"].to_numpy(float)
        idx = rng.integers(0, h.size, size=(n_boot, h.size))
        hb, mb = h[idx].sum(axis=1), m[idx].sum(axis=1)
        rate = np.divide(mb, hb, out=np.full(n_boot, np.nan), where=hb > 0)
        lo, hi = (np.nan, np.nan) if np.isnan(rate).all() else np.nanpercentile(rate, [2.5, 97.5])
        out.append({**dict(zip(cfg_cols + by, key)), "n_defects": h.size, "n_hit_defects": int((h > 0).sum()),
                    "miss_rate": m.sum() / h.sum() if h.sum() else np.nan, "boot_lo": lo, "boot_hi": hi})
    return pd.DataFrame(out)


def missed_depth(decisions: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """How far below the threshold the missed hits lie."""
    cfg_cols = ["regime", "proxy", "tau", "alpha", "calibration"]
    d = decisions.copy()
    y = df["e_relaxed"].to_numpy()[d["row"].to_numpy()]
    d["depth"] = d["tau"].to_numpy() - y
    d["hit"] = d["depth"] >= 0
    hits = d[d["hit"]]
    missed = hits[~hits["keep"]]
    g_hit = hits.groupby(cfg_cols, observed=True, dropna=False)["depth"]
    g_miss = missed.groupby(cfg_cols, observed=True, dropna=False)["depth"]
    out = pd.DataFrame({
        "median_depth_hits": g_hit.median(), "median_depth_missed": g_miss.median(),
        "frac_missed_within_0p5": g_miss.apply(lambda s: float((s < 0.5).mean())),
        "frac_missed_deeper_1": g_miss.apply(lambda s: float((s > 1.0).mean())),
        "miss_rate_deeper_1": g_miss.apply(lambda s: int((s > 1.0).sum())) / g_hit.apply(lambda s: int((s > 1.0).sum())),
    })
    return out.reset_index()
