"""Decision-level metrics for a prescreening rule."""
from __future__ import annotations

import numpy as np
import pandas as pd


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def screening_metrics(y: np.ndarray, keep: np.ndarray, tau: float, cost: np.ndarray | None = None) -> dict:
    """Summarise one screening decision.

    ``miss_rate`` is the fraction of true hits (``y <= tau``) that the rule
    discards; ``discard_frac`` is the fraction of all candidates that never
    reach a full relaxation. ``cost`` (ionic steps per candidate) gives the
    share of relaxation effort avoided.
    """
    y = np.asarray(y, dtype=float)
    keep = np.asarray(keep, dtype=bool)
    hit = y <= tau
    n, n_hit = y.size, int(hit.sum())
    missed = int((hit & ~keep).sum())
    lo, hi = wilson_interval(missed, n_hit)
    out = {
        "n": n,
        "n_hit": n_hit,
        "n_missed": missed,
        "miss_rate": missed / n_hit if n_hit else np.nan,
        "miss_lo": lo,
        "miss_hi": hi,
        "discard_frac": float((~keep).mean()) if n else np.nan,
        "precision": float(hit[keep].mean()) if keep.any() else np.nan,
    }
    if cost is not None:
        cost = np.asarray(cost, dtype=float)
        out["steps_saved_frac"] = float(cost[~keep].sum() / cost.sum()) if cost.sum() > 0 else np.nan
    return out


def metrics_by_cell(df: pd.DataFrame, keep: np.ndarray, tau: float, by: list[str], y_col: str = "e_relaxed") -> pd.DataFrame:
    rows = []
    keep = np.asarray(keep, dtype=bool)
    for key, idx in df.groupby(by, sort=False).indices.items():
        sub = df.iloc[idx]
        m = screening_metrics(sub[y_col].to_numpy(), keep[idx], tau, sub["n_frames"].to_numpy())
        key = key if isinstance(key, tuple) else (key,)
        rows.append({**dict(zip(by, key)), **m})
    return pd.DataFrame(rows)
