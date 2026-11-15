"""Minimal CIF reading and displacement statistics between two relaxation frames.

Used as a quality check: a relaxation that only rearranges the neighbourhood
of a defect moves few atoms, whereas a starting frame placed in an unrelaxed
host makes almost every atom in the supercell move.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_CELL_KEYS = ("length_a", "length_b", "length_c", "angle_alpha", "angle_beta", "angle_gamma")


def read_cif(path: str | Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return (lattice matrix in Å, species, fractional coordinates) of a P1 CIF."""
    text = Path(path).read_text()
    cell = {}
    for key in _CELL_KEYS:
        m = re.search(rf"_cell_{key}\s+([-\d.]+)", text)
        if m is None:
            raise ValueError(f"{path}: missing _cell_{key}")
        cell[key] = float(m.group(1))
    a, b, c = cell["length_a"], cell["length_b"], cell["length_c"]
    al, be, ga = np.radians([cell["angle_alpha"], cell["angle_beta"], cell["angle_gamma"]])
    va = np.array([a, 0.0, 0.0])
    vb = np.array([b * np.cos(ga), b * np.sin(ga), 0.0])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    vc = np.array([cx, cy, np.sqrt(max(c**2 - cx**2 - cy**2, 0.0))])
    lattice = np.vstack([va, vb, vc])

    cols, rows, in_loop = [], [], False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("loop_"):
            if in_loop and rows:
                break
            cols, in_loop = [], False
            continue
        if s.startswith("_atom_site_"):
            cols.append(s)
            in_loop = True
            continue
        if in_loop and s and not s.startswith("_"):
            parts = s.split()
            if len(parts) >= len(cols):
                rows.append(parts)
    if not rows:
        raise ValueError(f"{path}: no atom sites found")
    ix = [cols.index(f"_atom_site_fract_{k}") for k in "xyz"]
    species_col = "_atom_site_type_symbol" if "_atom_site_type_symbol" in cols else "_atom_site_label"
    species = [re.match(r"[A-Z][a-z]?", r[cols.index(species_col)]).group(0) for r in rows]
    frac = np.array([[float(r[i]) for i in ix] for r in rows])
    return lattice, species, frac


def displacement_stats(first: str | Path, last: str | Path, threshold: float = 0.1) -> dict:
    """Atomic displacements between two frames with the same atom ordering."""
    lat0, sp0, f0 = read_cif(first)
    lat1, sp1, f1 = read_cif(last)
    if sp0 != sp1:
        raise ValueError(f"atom order differs between {first} and {last}")
    d = f1 - f0
    d -= np.round(d)
    u = np.linalg.norm(d @ lat0, axis=1)
    strain = float(np.max(np.abs(lat1 - lat0)))
    return {
        "n_atoms": int(u.size),
        "u_median": float(np.median(u)),
        "u_max": float(u.max()),
        "frac_moved": float(np.mean(u > threshold)),
        "cell_change": strain,
    }
