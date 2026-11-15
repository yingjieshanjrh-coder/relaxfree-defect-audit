"""Secondary analyses: what drives the relaxation energy, and charged-state re-relaxation."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

from relaxfree.dataset import load_neutral
from relaxfree.features import identity_features
from relaxfree.models import make_model
from relaxfree.screening import screening_metrics

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

FEATURE_BLOCKS = {
    "host species": lambda c: c.startswith(("A_", "B_", "host_", "family_")),
    "placed species": lambda c: c.startswith("X_") and c not in ("X_size_ratio", "X_shell_d_en"),
    "removed species": lambda c: c.startswith("R_"),
    "placed-removed contrast": lambda c: c.startswith("XR_"),
    "size ratio / shell contrast": lambda c: c in ("X_size_ratio", "X_shell_d_en"),
    "site and class": lambda c: c.startswith(("site_", "class_", "has_")),
}


def relaxation_drivers(df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Blockwise permutation importance for predicting the relaxation energy on unseen hosts."""
    X, y = feats.to_numpy(dtype=float), df["e_relax"].to_numpy()
    cols = list(feats.columns)
    rng = np.random.default_rng(0)
    rows = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, df["host"])):
        reg = make_model("hgb", fold).fit(X[tr], y[tr])
        base = np.mean(np.abs(reg.predict(X[te]) - y[te]))
        for block, pick in FEATURE_BLOCKS.items():
            idx = [i for i, c in enumerate(cols) if pick(c)]
            deltas = []
            for _ in range(10):
                Xp = X[te].copy()
                Xp[:, idx] = Xp[rng.permutation(len(te))][:, idx]
                deltas.append(np.mean(np.abs(reg.predict(Xp) - y[te])) - base)
            rows.append({"fold": fold, "block": block, "mae_increase": float(np.mean(deltas)), "base_mae": base})
    out = pd.DataFrame(rows).groupby("block").agg(mae_increase=("mae_increase", "mean"),
                                                  sd=("mae_increase", "std"), base_mae=("base_mae", "mean"))
    return out.sort_values("mae_increase", ascending=False).reset_index()


def size_trends(df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = df.assign(size_ratio=feats["X_size_ratio"], mismatch=feats["XR_size_mismatch"].abs())
    for cls, col in [("int_impurity", "size_ratio"), ("self_interstitial", "size_ratio"),
                     ("sub_impurity", "mismatch"), ("antisite", "mismatch")]:
        sub = d[d["defect_class"] == cls].dropna(subset=[col])
        rho = spearmanr(sub[col], sub["e_relax"])
        rows.append({"defect_class": cls, "descriptor": col, "n": len(sub), "spearman": rho.statistic, "p": rho.pvalue})
    return pd.DataFrame(rows)


def charged_states(ch: pd.DataFrame) -> pd.DataFrame:
    """Charged runs start from the neutral relaxed geometry, so their first frame is a
    single point at that geometry. Does it suffice to screen the charged state?"""
    rows = []
    for (q, inter), sub in ch.groupby(["charge", "interstitial"]):
        base = {"charge": q, "interstitial": inter, "median_e_relax": sub["e_relax"].median(),
                "q90_e_relax": sub["e_relax"].quantile(0.9), "frac_gt_0p5": (sub["e_relax"] > 0.5).mean()}
        for tau in (2.0, 3.0, 4.0):
            m = screening_metrics(sub["e_relaxed"], sub["e_unrelaxed"] <= tau, tau)
            rows.append({**base, "tau": tau, **m})
    return pd.DataFrame(rows)


def main() -> None:
    df = load_neutral(ROOT / "data" / "processed" / "defects_neutral.csv")
    feats = identity_features(df)
    drivers = relaxation_drivers(df, feats)
    drivers.to_csv(RES / "relaxation_drivers.csv", index=False, float_format="%.4f")
    trends = size_trends(df, feats)
    trends.to_csv(RES / "relaxation_size_trends.csv", index=False, float_format="%.4g")
    ch = load_neutral(ROOT / "data" / "processed" / "defects_charged.csv")
    charged = charged_states(ch)
    charged.to_csv(RES / "charged_state_screen.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 200)
    print(drivers.round(3).to_string(index=False))
    print(trends.round(4).to_string(index=False))
    cols = ["charge", "interstitial", "n", "median_e_relax", "q90_e_relax", "n_hit", "miss_rate"]
    print(charged[charged["tau"] == 3.0][cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
