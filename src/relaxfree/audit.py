"""Repeated calibrate-and-screen experiments."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .conformal import grouped_recall_margin, keep_mask, recall_margin
from .proxies import NEEDS_SINGLE_POINT, PROXIES, fit_predict
from .screening import wilson_interval
from .splits import make_splits

# calibration name -> column whose values define the calibration groups
CALIBRATION_GROUPS = {"classwise": "defect_class", "cellwise": "cell"}
CALIBRATIONS = ["none", "pooled", *CALIBRATION_GROUPS]


@dataclass
class AuditConfig:
    regimes: list[str] = field(default_factory=lambda: ["random", "host", "family"])
    proxies: list[str] = field(default_factory=lambda: list(PROXIES))
    taus: list[float] = field(default_factory=lambda: [2.0, 3.0, 4.0])
    alphas: list[float] = field(default_factory=lambda: [0.05, 0.1, 0.2])
    n_repeats: dict = field(default_factory=lambda: {"random": 50, "host": 5, "family": 10})
    model: str = "hgb"
    seed: int = 7


def _splits_with_repeat(regime: str, df: pd.DataFrame, n_repeats: int, seed: int):
    per_repeat = {"random": 1, "host": df["host"].nunique(), "family": df["family"].nunique()}[regime]
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
                        for calib in CALIBRATIONS:
                            if calib == "none":
                                if alpha != cfg.alphas[0]:
                                    continue
                                keep = p_test <= tau
                                finite = np.ones(test.size, dtype=bool)
                            elif calib == "pooled":
                                m = recall_margin(p_cal, y[cal], tau, alpha)
                                keep = keep_mask(p_test, m, tau)
                                finite = np.full(test.size, np.isfinite(m))
                            else:
                                g = groups[CALIBRATION_GROUPS[calib]]
                                m = grouped_recall_margin(p_cal, y[cal], g[cal], tau, alpha)
                                keep = keep_mask(p_test, m, tau, g[test])
                                finite = np.array([np.isfinite(m.get(k, np.inf)) for k in g[test]])
                            chunks.append(pd.DataFrame({
                                "regime": regime, "rep": rep, "proxy": proxy, "tau": tau,
                                "alpha": np.nan if calib == "none" else alpha, "calibration": calib,
                                "row": test, "keep": keep, "calibrated": finite, "abs_err": np.abs(p_test - y[test]),
                            }))
    out = pd.concat(chunks, ignore_index=True)
    for col in ("regime", "proxy", "calibration"):
        out[col] = out[col].astype("category")
    return out


def summarise(decisions: pd.DataFrame, df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Aggregate decisions over repeats into miss rate, discard fraction and effort saved."""
    cfg_cols = ["regime", "proxy", "tau", "alpha", "calibration"]
    d = decisions.copy()
    rows = d["row"].to_numpy()
    y = df["e_relaxed"].to_numpy()[rows]
    steps = df["n_frames"].to_numpy()[rows].astype(float)
    sp = d["proxy"].map(NEEDS_SINGLE_POINT).astype(float).to_numpy()
    for col in by:
        d[col] = df[col].to_numpy()[rows]
    d["hit"] = y <= d["tau"].to_numpy()
    d["missed"] = d["hit"] & ~d["keep"]
    d["discard"] = ~d["keep"]
    d["steps"] = steps
    d["steps_saved"] = np.where(d["discard"], steps - sp, 0.0)
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
