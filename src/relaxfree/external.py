"""Reduction of the external chalcogenide vacancy relaxations to one row per defect."""
from __future__ import annotations

import csv
import json
import re

import numpy as np
import pandas as pd

csv.field_size_limit(10**9)

_SEGMENT_TIME = re.compile(r"vasprun\.xml_(\d+)_(\d+)_(\d+)on(\d+)_(\d+)_(\d+)\.gz")
_DEFECT = re.compile(r"v_([A-Z][a-z]?)_s(\d+)_(-?\d+)$")

# a relaxation restarted from its own CONTCAR continues at the same energy;
# larger jumps mean a changed setting or a different structure
MAX_JOIN = 0.1  # eV
# energy rising along a relaxation signals electronic trouble
MAX_RISE = 0.05  # eV


def segment_order(name: str) -> tuple:
    """Archived segments carry their start time (HH_MM_SSonDD_MM_YY); the plain file is the latest."""
    m = _SEGMENT_TIME.match(name)
    if not m:
        return (99,) * 6
    hh, mi, ss, dd, mo, yy = map(int, m.groups())
    return (yy, mo, dd, hh, mi, ss)


def join_segments(seg: pd.DataFrame) -> dict:
    """Concatenate the segments of one relaxation and record quality indicators."""
    seg = seg.assign(_t=seg["file"].map(segment_order)).sort_values("_t")
    seg = seg[seg["n_steps"] > 0]
    nelm = seg["nelm"].dropna().max() if len(seg) else np.nan
    energies, nsc, joins, last = [], [], [], None
    for _, r in seg.iterrows():
        e = json.loads(r["energies"])
        if last is not None and e:
            joins.append(e[0] - last)
        energies += e
        nsc += json.loads(r["n_scsteps"])
        last = e[-1] if e else last
    e = np.array(energies, dtype=float)
    finite = e[np.isfinite(e)]
    unconverged = [bool(nelm) and n >= nelm for n in nsc]
    return {
        "n_segments": int(len(seg)),
        "n_steps": int(e.size),
        "e_first": float(e[0]) if e.size else np.nan,
        "e_last": float(e[-1]) if e.size else np.nan,
        "max_join": float(np.max(np.abs(joins))) if joins else 0.0,
        "max_rise": float(np.max(np.diff(finite))) if finite.size > 1 else 0.0,
        "n_unconverged": int(sum(unconverged)),
        "ends_converged": bool(unconverged) and not unconverged[0] and not unconverged[-1],
        "final_complete": bool(seg["complete"].iloc[-1]) if len(seg) else False,
        "has_nan": bool(np.isnan(e).any()),
    }


def summarise_segments(segments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subset, host, defect), g in segments.groupby(["top", "host", "defect"]):
        m = _DEFECT.match(defect)
        row = {"subset": subset, "host": host, "defect": defect,
               "element": m.group(1) if m else "", "charge": int(m.group(3)) if m else np.nan,
               "n_distortions": int(g["distortion"].nunique())}
        finals = {}
        for dist, gd in g.groupby("distortion"):
            tr = join_segments(gd)
            if tr["final_complete"] and tr["ends_converged"] and np.isfinite(tr["e_last"]):
                finals[dist] = tr["e_last"]
            if dist == "Unperturbed":
                row.update({f"unp_{k}": v for k, v in tr.items()})
        row["e_best_final"] = min(finals.values()) if finals else np.nan
        row["best_distortion"] = min(finals, key=finals.get) if finals else ""
        rows.append(row)
    out = pd.DataFrame(rows)
    out["e_relax"] = out["unp_e_first"] - out["unp_e_last"]
    out["e_reconstruction"] = out["unp_e_last"] - out["e_best_final"]
    usable = out["unp_final_complete"].fillna(False).astype(bool) & out["unp_ends_converged"].fillna(False).astype(bool) \
        & ~out["unp_has_nan"].fillna(True).astype(bool) & (out["unp_max_join"] < MAX_JOIN)
    out["usable"] = usable
    out["clean"] = usable & (out["unp_n_unconverged"] == 0) & (out["unp_max_rise"] < MAX_RISE)
    return out


def load_segments(path: str) -> pd.DataFrame:
    cols = ["top", "host", "defect", "distortion", "file", "n_steps", "complete", "nelm", "energies", "n_scsteps"]
    return pd.read_csv(path, usecols=cols)
