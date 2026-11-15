"""Parsing of the upstream label files into one row per relaxation trajectory.

Each upstream CSV row is ``<structure file>,<defect formation energy>``, where
the structure file is one ionic step (``POSCARk``) of a DFT relaxation. Step 0
is the ideal-site (unrelaxed) defect and the last step is the relaxed one, so a
trajectory gives both the unrelaxed and the relaxed formation energy of the
same defect at the same level of theory.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

GROUP_IV = {"C", "Si", "Ge", "Sn"}
GROUP_III = {"B", "Al", "Ga", "In"}
GROUP_II = {"Zn", "Cd"}

CHARGE_DIRS = {"Neutral": 0, "Charged+1": 1, "Charged-1": -1, "Charged+2": 2, "Charged-2": -2}

# frames above this energy were removed upstream for the impurity set; the
# same cap is applied to native defects so both subsets are censored alike
ENERGY_CAP = 15.0

_NATIVE_RE = re.compile(r"^(?P<idx>\d+)-(?P<host>[A-Za-z]+)-(?P<defect>.+?)-POSCAR(?P<step>\d+)(?:-[A-Za-z0-9+\-]+)?\.cif$")
_IMPURITY_RE = re.compile(
    r"^(?P<idx>\d+)-(?P<host>[A-Za-z]+)-(?P<site>M_A|M_B|M_i_A|M_i_B|M_i_neut)-(?P<element>[A-Z][a-z]?)"
    r"-POSCAR(?P<step>\d+)(?:-[A-Za-z0-9+\-]+)?\.cif$"
)
_SITE_MAP = {"M_A": "A", "M_B": "B", "M_i_A": "i_A", "M_i_B": "i_B", "M_i_neut": "i_neut"}


def host_species(host: str) -> tuple[str, str]:
    """Return the (A, B) species of a zinc blende host; elemental hosts give A == B."""
    el = re.findall(r"[A-Z][a-z]?", host)
    if len(el) == 1:
        return el[0], el[0]
    if len(el) == 2:
        return el[0], el[1]
    raise ValueError(f"unexpected host formula {host!r}")


def host_family(host: str) -> str:
    a, b = host_species(host)
    if a in GROUP_IV and b in GROUP_IV:
        return "IV"
    if a in GROUP_III:
        return "III-V"
    if a in GROUP_II:
        return "II-VI"
    raise ValueError(f"cannot assign family to {host!r}")


def parse_native_defect(host: str, name: str) -> dict:
    """Decode native defect labels such as ``V_Zn``, ``Ga_N`` or ``As_i_neut``."""
    a, b = host_species(host)
    m = re.fullmatch(r"([A-Z][a-z]?)_i_(A|B|neut)", name)
    if m:
        return {"defect_class": "self_interstitial", "site": f"i_{m.group(2)}", "element": m.group(1), "replaced": None}
    m = re.fullmatch(r"V_([A-Z][a-z]?)", name)
    if m:
        removed = m.group(1)
        if removed not in (a, b):
            raise ValueError(f"vacancy {name} does not belong to {host}")
        return {"defect_class": "vacancy", "site": "A" if removed == a else "B", "element": None, "replaced": removed}
    m = re.fullmatch(r"([A-Z][a-z]?)_([A-Z][a-z]?)", name)
    if m:
        placed, removed = m.groups()
        if removed not in (a, b) or placed not in (a, b):
            raise ValueError(f"antisite {name} does not belong to {host}")
        return {"defect_class": "antisite", "site": "A" if removed == a else "B", "element": placed, "replaced": removed}
    raise ValueError(f"unrecognised native defect label {name!r}")


def _read_frames(path: Path, native: bool) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, names=["file", "dfe"])
    pattern = _NATIVE_RE if native else _IMPURITY_RE
    parts = raw["file"].str.extract(pattern)
    bad = parts["step"].isna()
    if bad.any():
        raise ValueError(f"{bad.sum()} unparsed file names in {path}, e.g. {raw.loc[bad, 'file'].iloc[0]}")
    frames = pd.concat([raw, parts], axis=1)
    frames["step"] = frames["step"].astype(int)
    return frames


def _describe(frames: pd.DataFrame, native: bool) -> pd.DataFrame:
    keys = ["idx", "host", "defect"] if native else ["idx", "host", "site", "element"]
    frames = frames.sort_values(keys + ["step"])
    traj = frames.groupby(keys, sort=False).agg(
        first_step=("step", "min"),
        last_step=("step", "max"),
        n_frames=("step", "size"),
        e_unrelaxed=("dfe", "first"),
        e_relaxed=("dfe", "last"),
        e_min=("dfe", "min"),
        e_max=("dfe", "max"),
    ).reset_index()
    if native:
        info = pd.DataFrame([parse_native_defect(h, d) for h, d in zip(traj["host"], traj["defect"])])
        traj = pd.concat([traj.drop(columns="defect"), info], axis=1)
    else:
        traj["site"] = traj["site"].map(_SITE_MAP)
        sub = traj["site"].isin(["A", "B"])
        species = traj["host"].map(host_species)
        traj["replaced"] = np.where(
            sub,
            [s[0] if site == "A" else s[1] for s, site in zip(species, traj["site"])],
            None,
        )
        traj["defect_class"] = np.where(sub, "sub_impurity", "int_impurity")
    traj["source"] = "native" if native else "impurity"
    return traj


def load_trajectories(label_root: str | Path) -> pd.DataFrame:
    """Read every label file under ``label_root`` and return one row per trajectory."""
    label_root = Path(label_root)
    tables = []
    for path in sorted(label_root.rglob("id_prop_*.csv")):
        native = "native" in path.parts
        mu = "A_rich" if "A_rich" in path.parts else "B_rich"
        charge = CHARGE_DIRS[path.parent.name]
        traj = _describe(_read_frames(path, native), native)
        traj["charge"] = charge
        traj["mu"] = mu
        tables.append(traj)
    out = pd.concat(tables, ignore_index=True)
    a_b = out["host"].map(host_species)
    out["host_A"] = [x[0] for x in a_b]
    out["host_B"] = [x[1] for x in a_b]
    out["family"] = out["host"].map(host_family)
    out["e_relax"] = out["e_unrelaxed"] - out["e_relaxed"]
    out["interstitial"] = out["defect_class"].isin(["self_interstitial", "int_impurity"])
    return out


def quality_flags(traj: pd.DataFrame, cap: float = ENERGY_CAP) -> pd.DataFrame:
    """Attach boolean QC columns; ``keep`` marks trajectories used in the main analysis."""
    t = traj.copy()
    t["flag_no_initial_frame"] = t["first_step"] > 0
    t["flag_above_cap"] = t["e_unrelaxed"] > cap
    t["flag_corrupt"] = (t["e_min"] < -10.0) | (t["e_relax"] < -0.1)
    t["flag_single_frame"] = t["n_frames"] < 2
    t["censored"] = t["flag_no_initial_frame"] | t["flag_above_cap"]
    t["keep"] = ~(t["censored"] | t["flag_corrupt"] | t["flag_single_frame"])
    return t
