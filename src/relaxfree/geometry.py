"""Geometry of the first recorded frame of each trajectory.

The audit treats step 0 as the defect on its ideal site. That is only true if
every atom of the first frame sits on a site of the ideal zinc blende supercell.
This module measures the largest atomic displacement from the ideal sites so
that trajectories started from pre-displaced geometries can be told apart.
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .parse import _IMPURITY_RE, _NATIVE_RE, _SITE_MAP, parse_native_defect

# ideal sites of the 2x2x2 conventional cell (and tetrahedral interstitial sites)
# have fractional coordinates that are multiples of 1/16 for either origin choice
GRID = 16
IDEAL_TOL = 0.02  # Angstrom

_ATOM = re.compile(r"^\s*[A-Z][a-z]?\s+\S+\s+\d+\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+[\d.]+\s*$")
_CELL = re.compile(r"_cell_length_a\s+([\d.]+)")

ARCHIVES = {
    "native": ("Dataset_1/Dataset_1.zip", "Dataset_1/A_rich/Neutral/", _NATIVE_RE),
    "impurity": ("Dataset_2/A_rich/Neutral.zip", "Neutral/", _IMPURITY_RE),
}


def displacement_from_ideal(cif_text: str) -> tuple[int, float]:
    """Return (number of displaced atoms, largest displacement in Angstrom) for a cubic supercell."""
    a = float(_CELL.search(cif_text).group(1))
    frac = np.array([[float(v) for v in m.groups()] for m in map(_ATOM.match, cif_text.splitlines()) if m])
    scaled = frac * GRID
    disp = np.linalg.norm(np.abs(scaled - np.round(scaled)) / GRID * a, axis=1)
    return int((disp > IDEAL_TOL).sum()), float(disp.max())


def first_frame_geometry(upstream_root: str | Path) -> pd.DataFrame:
    """Displacement statistics of the first frame of every neutral A-rich trajectory."""
    upstream_root = Path(upstream_root)
    rows = []
    for source, (archive, prefix, pattern) in ARCHIVES.items():
        with zipfile.ZipFile(upstream_root / archive) as zf:
            for name in zf.namelist():
                if not name.startswith(prefix) or "__MACOSX" in name:
                    continue
                m = pattern.match(Path(name).name)
                if not m or m["step"] != "0":
                    continue
                if source == "native":
                    info = parse_native_defect(m["host"], m["defect"])
                    site, element = info["site"], info["element"] or ""
                else:
                    site, element = _SITE_MAP[m["site"]], m["element"]
                n_disp, max_disp = displacement_from_ideal(zf.read(name).decode())
                rows.append({"source": source, "idx": m["idx"], "host": m["host"], "site": site, "element": element,
                             "start_n_displaced": n_disp, "start_max_disp": max_disp})
    return pd.DataFrame(rows)
