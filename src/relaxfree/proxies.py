"""The three screening proxies compared in the audit.

unrelaxed  formation energy of the ideal-site defect (one SCF, no model)
identity   regression on identity descriptors only (no DFT on the defect)
hybrid     unrelaxed energy minus a learned relaxation energy (one SCF + model)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import make_model

PROXIES = ["unrelaxed", "identity", "hybrid"]
NEEDS_SINGLE_POINT = {"unrelaxed": True, "identity": False, "hybrid": True}


def proxy_cost(proxy: str) -> int:
    """Ionic steps spent on a candidate before the keep/discard decision.

    ``step<k>`` is the formation energy after k ionic steps of the relaxation,
    which costs the single-point step plus k further steps.
    """
    if proxy.startswith("step"):
        return int(proxy[4:]) + 1
    return int(NEEDS_SINGLE_POINT[proxy])


def fit_predict(
    proxy: str, df: pd.DataFrame, feats: pd.DataFrame, train: np.ndarray, target_idx: np.ndarray,
    model: str = "hgb", seed: int = 0,
) -> np.ndarray:
    """Predict the relaxed formation energy of ``target_idx`` rows using ``train`` rows."""
    e_unrel = df["e_unrelaxed"].to_numpy()
    if proxy == "unrelaxed":
        return e_unrel[target_idx]
    if proxy.startswith("step"):
        return df[f"e_{proxy}"].to_numpy()[target_idx]
    X = feats.to_numpy(dtype=float)
    if proxy == "identity":
        reg = make_model(model, seed).fit(X[train], df["e_relaxed"].to_numpy()[train])
        return reg.predict(X[target_idx])
    if proxy == "hybrid":
        Xh = np.column_stack([X, e_unrel])
        reg = make_model(model, seed).fit(Xh[train], df["e_relax"].to_numpy()[train])
        return e_unrel[target_idx] - np.clip(reg.predict(Xh[target_idx]), 0.0, None)
    raise ValueError(f"unknown proxy {proxy!r}")
