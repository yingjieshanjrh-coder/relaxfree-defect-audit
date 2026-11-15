"""Descriptors that need no defect geometry: only host, site and species identity."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ELEMENT_PROPS = ["en", "rcov", "row", "col", "ie1", "ea", "mendeleev", "nval", "polariz", "vol_pa"]
SITES = ["A", "B", "i_A", "i_B", "i_neut"]
CLASSES = ["vacancy", "antisite", "sub_impurity", "self_interstitial", "int_impurity"]
FAMILIES = ["IV", "III-V", "II-VI"]

_DEFAULT_TABLE = Path(__file__).resolve().parents[2] / "data" / "external" / "elements.csv"


def load_elements(path: str | Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or _DEFAULT_TABLE).set_index("symbol")


def _props(elements: pd.DataFrame, symbols: pd.Series, prefix: str) -> pd.DataFrame:
    """Elemental properties for a column of symbols; empty symbols give NaN."""
    block = elements[ELEMENT_PROPS].reindex(symbols.where(symbols != "", None).to_numpy())
    block.index = symbols.index
    return block.add_prefix(prefix)


def identity_features(df: pd.DataFrame, elements: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the relaxation-free feature matrix.

    Nothing here depends on a supercell calculation. Missing species (no atom
    placed for a vacancy, none removed for an interstitial) are left as NaN and
    flagged, so tree models can split on them directly.
    """
    elements = load_elements() if elements is None else elements
    a = _props(elements, df["host_A"], "A_")
    b = _props(elements, df["host_B"], "B_")
    x = _props(elements, df["element"], "X_")
    r = _props(elements, df["replaced"], "R_")

    feats = pd.concat([a, b, x, r], axis=1)
    for p in ("en", "rcov", "nval", "col"):
        feats[f"host_d_{p}"] = a[f"A_{p}"] - b[f"B_{p}"]
        feats[f"XR_d_{p}"] = x[f"X_{p}"] - r[f"R_{p}"]
    bond = a["A_rcov"] + b["B_rcov"]
    feats["host_bond"] = bond
    feats["host_mean_en"] = 0.5 * (a["A_en"] + b["B_en"])
    feats["X_size_ratio"] = x["X_rcov"] / bond
    feats["XR_size_mismatch"] = (x["X_rcov"] - r["R_rcov"]) / bond
    # mean electronegativity contrast with the nearest-neighbour shell of the ideal site
    shell_en = np.select(
        [df["site"].isin(["A", "i_B"]), df["site"].isin(["B", "i_A"])],
        [b["B_en"], a["A_en"]],
        default=feats["host_mean_en"],
    )
    feats["X_shell_d_en"] = x["X_en"] - shell_en

    feats["has_X"] = (df["element"] != "").astype(float)
    feats["has_R"] = (df["replaced"] != "").astype(float)
    feats["charge"] = df["charge"].astype(float)
    for s in SITES:
        feats[f"site_{s}"] = (df["site"] == s).astype(float)
    for c in CLASSES:
        feats[f"class_{c}"] = (df["defect_class"] == c).astype(float)
    for f in FAMILIES:
        feats[f"family_{f}"] = (df["family"] == f).astype(float)
    return feats
