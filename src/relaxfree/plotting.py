"""Shared matplotlib styling."""
from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"

FAMILY_COLORS = {"IV": "#2a78d6", "III-V": "#eb6834", "II-VI": "#138a60"}
PROXY_COLORS = {"unrelaxed": "#2a78d6", "identity": "#eb6834", "hybrid": "#138a60"}
PROXY_LABELS = {
    "unrelaxed": "unrelaxed energy",
    "identity": "identity-only model",
    "hybrid": "unrelaxed + learned relaxation",
}
PROXY_MARKERS = {"unrelaxed": "o", "identity": "s", "hybrid": "^"}
REGIME_LABELS = {"random": "seen hosts", "host": "unseen host", "family": "unseen family"}
CLASS_LABELS = {
    "vacancy": "vacancy",
    "antisite": "antisite",
    "sub_impurity": "substitutional\nimpurity",
    "self_interstitial": "self-\ninterstitial",
    "int_impurity": "interstitial\nimpurity",
}

BLUES = LinearSegmentedColormap.from_list(
    "audit_blues", ["#f3f7fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]
)


def use_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "axes.edgecolor": INK_SOFT,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "pdf.fonttype": 42,
    })


def panel_label(ax, text: str) -> None:
    ax.text(-0.02, 1.06, text, transform=ax.transAxes, fontsize=10, fontweight="bold",
            color=INK, ha="right", va="bottom")
