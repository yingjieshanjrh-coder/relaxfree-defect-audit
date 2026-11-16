"""Assembly of the analysis tables from parsed trajectories."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import IDEAL_TOL, first_frame_geometry
from .parse import PARTIAL_STEPS, load_trajectories, quality_flags

ID_COLS = ["host", "site", "element", "replaced", "charge"]
CLASS_ORDER = ["vacancy", "antisite", "sub_impurity", "self_interstitial", "int_impurity"]
FAMILY_ORDER = ["IV", "III-V", "II-VI"]

COLUMNS = [
    "defect_id", "host", "family", "host_A", "host_B", "defect_class", "interstitial", "site",
    "element", "replaced", "charge", "source", "n_frames", "e_unrelaxed", "e_relaxed", "e_min", "e_relax",
    "start_n_displaced", "start_max_disp", "ideal_start",
]


def _relabel_host_species(traj: pd.DataFrame) -> pd.DataFrame:
    """Impurity rows whose 'impurity' is a host species are native defects."""
    t = traj.copy()
    is_host = (t["source"] == "impurity") & (t["element"].eq(t["host_A"]) | t["element"].eq(t["host_B"]))
    sub = t["site"].isin(["A", "B"])
    same_site = sub & t["element"].eq(t["replaced"])
    t = t[~(is_host & same_site)].copy()
    is_host = is_host.reindex(t.index)
    sub = sub.reindex(t.index)
    t.loc[is_host & sub, "defect_class"] = "antisite"
    t.loc[is_host & ~sub, "defect_class"] = "self_interstitial"
    return t


def _attach_start_geometry(traj: pd.DataFrame, upstream_root: str | Path) -> pd.DataFrame:
    """Add first-frame displacement columns to the neutral trajectories."""
    geom = first_frame_geometry(upstream_root)
    t = traj.assign(_el=traj["element"].fillna(""))
    geom = geom.rename(columns={"element": "_el"})
    t = t.merge(geom, on=["source", "idx", "host", "site", "_el"], how="left").drop(columns="_el")
    neutral = t["charge"] == 0
    for col in ("start_n_displaced", "start_max_disp"):
        t.loc[~neutral, col] = np.nan
    t["ideal_start"] = t["start_max_disp"] <= IDEAL_TOL
    return t


def _defect_id(t: pd.DataFrame) -> pd.Series:
    element = t["element"].fillna("")
    return (
        t["host"] + ":" + t["site"] + ":" + element.where(element != "", "vac")
        + ":" + t["charge"].map("{:+d}".format)
    )


def build_tables(label_root: str | Path, upstream_root: str | Path) -> dict[str, pd.DataFrame]:
    """Return the neutral table, the charged-state table and a QC summary."""
    every = quality_flags(load_trajectories(label_root))
    b_rich = every[(every["mu"] == "B_rich") & (every["charge"] == 0) & every["keep"]].copy()
    b_rich["defect_id"] = _defect_id(b_rich)
    b_rich = b_rich.drop_duplicates("defect_id")[["defect_id", "e_unrelaxed", "e_relaxed"]]
    traj = every[every["mu"] == "A_rich"]
    traj = _attach_start_geometry(traj, upstream_root)
    traj = _relabel_host_species(traj)

    qc = (
        traj.assign(neutral=traj["charge"] == 0)
        .groupby(["neutral", "source"])[
            ["flag_no_initial_frame", "flag_above_cap", "flag_corrupt", "flag_single_frame", "keep"]
        ]
        .sum()
    )
    qc["total"] = traj.assign(neutral=traj["charge"] == 0).groupby(["neutral", "source"]).size()

    kept = traj[traj["keep"]].copy()
    # the same defect can appear in both upstream subsets; prefer the native entry
    kept["_prio"] = (kept["source"] != "native").astype(int)
    kept = kept.sort_values(ID_COLS + ["_prio"]).drop_duplicates(ID_COLS, keep="first")
    kept["element"] = kept["element"].fillna("")
    kept["replaced"] = kept["replaced"].fillna("")
    kept["defect_id"] = _defect_id(kept)
    kept = kept.sort_values(["family", "host", "defect_class", "site", "element", "charge"])
    steps = kept.loc[kept["charge"] == 0, ["defect_id"] + [f"e_step{k}" for k in PARTIAL_STEPS]]
    kept = kept[COLUMNS]

    censored = traj[traj["censored"] & ~traj["flag_corrupt"] & (traj["charge"] == 0)].copy()
    single = traj[
        traj["flag_single_frame"] & ~traj["censored"] & ~traj["flag_corrupt"] & (traj["charge"] == 0)
    ].copy()
    return {
        "single_frame": single.reset_index(drop=True),
        "neutral": kept[kept["charge"] == 0].reset_index(drop=True),
        "charged": kept[kept["charge"] != 0].reset_index(drop=True),
        "censored": censored.reset_index(drop=True),
        "qc": qc.reset_index(),
        "steps": steps.reset_index(drop=True),
        "b_rich": b_rich.reset_index(drop=True),
    }


def load_table(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    for col in ("element", "replaced"):
        df[col] = df[col].fillna("")
    return df


def load_neutral(path: str | Path, ideal_only: bool = True) -> pd.DataFrame:
    """Neutral defects; by default only those whose first frame is the ideal-site geometry."""
    df = load_table(path)
    if ideal_only:
        df = df[df["ideal_start"]].reset_index(drop=True)
    return df


def attach_partial_steps(df: pd.DataFrame, path: str | Path) -> pd.DataFrame:
    """Add the partially relaxed formation energies ``e_step<k>`` to a neutral table."""
    steps = pd.read_csv(path)
    out = df.merge(steps, on="defect_id", how="left", validate="one_to_one")
    if out[[c for c in steps.columns if c != "defect_id"]].isna().any().any():
        raise ValueError("missing partial-relaxation energies")
    return out
