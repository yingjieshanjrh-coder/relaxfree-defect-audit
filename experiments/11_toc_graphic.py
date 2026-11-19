"""Table-of-contents graphic (3.25 in x 1.75 in), drawn from the result files."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from relaxfree.dataset import load_neutral  # noqa: E402

INK, SOFT = "#0b0b0b", "#52514e"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "axes.linewidth": 0.5,
                     "axes.edgecolor": SOFT, "xtick.color": SOFT, "ytick.color": SOFT,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "pdf.fonttype": 42})

df = load_neutral(REPO / "data/processed/defects_neutral.csv")
naive = pd.read_csv(REPO / "results/naive_unrelaxed_screen.csv")
naive = naive[(naive["tau"] == 3.0) & (naive["family"] == "all")].iloc[0]
over = pd.read_csv(REPO / "results/screen_overall.csv")
pooled = over[(over["tau"] == 3.0) & (over["alpha"] == 0.1) & (over["calibration"] == "pooled")
              & (over["proxy"] == "unrelaxed") & (over["regime"] == "random")].iloc[0]

fig = plt.figure(figsize=(3.25, 1.75), dpi=600)
ax0 = fig.add_axes([0.11, 0.2, 0.36, 0.66])
hit = df["e_relaxed"] <= 3.0
missed = hit & (df["e_unrelaxed"] > 3.0)
ax0.scatter(df.loc[~missed, "e_relaxed"], df.loc[~missed, "e_unrelaxed"], s=1.2, color="#9ec5f4", lw=0, rasterized=True)
ax0.scatter(df.loc[missed, "e_relaxed"], df.loc[missed, "e_unrelaxed"], s=2.0, color=ORANGE, lw=0, rasterized=True)
ax0.plot([-4, 15], [-4, 15], color=SOFT, lw=0.5)
ax0.axvline(3, color=SOFT, lw=0.4, ls=(0, (2, 1.5)))
ax0.axhline(3, color=SOFT, lw=0.4, ls=(0, (2, 1.5)))
ax0.set(xlim=(-2, 12), ylim=(-2, 15))
ax0.set_xlabel("relaxed $E_\\mathrm{f}$ (eV)", labelpad=1)
ax0.set_ylabel("ideal-site $E_\\mathrm{f}$ (eV)", labelpad=1)
ax0.tick_params(length=2, pad=1)
for s in ("top", "right"):
    ax0.spines[s].set_visible(False)
ax0.text(-1.6, 14.6, "missed hits", color=ORANGE, va="top", fontsize=6)

ax1 = fig.add_axes([0.6, 0.2, 0.37, 0.66])
vals = [naive["miss_rate"], pooled["miss_rate"]]
ax1.bar([0, 1], vals, width=0.55, color=[ORANGE, BLUE])
ax1.axhline(0.1, color=SOFT, lw=0.5, ls=(0, (2, 1.5)))
ax1.text(0.5, 0.106, "target", color=SOFT, fontsize=5.5, va="bottom", ha="center")
for x, v in zip([0, 1], vals):
    ax1.text(x, v + 0.008, f"{v:.0%}", ha="center", va="bottom", fontsize=6.5, color=INK)
ax1.set_xticks([0, 1], ["skip\nrelaxation", "calibrated\nmargin"])
ax1.set_ylim(0, 0.3)
ax1.set_yticks([0, 0.1, 0.2, 0.3], ["0", "10%", "20%", "30%"])
ax1.set_ylabel("hits missed", labelpad=1)
ax1.tick_params(length=2, pad=1)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)
fig.text(0.5, 0.97, "Calibrated relaxation-free defect screening", ha="center", va="top", fontsize=7, color=INK, weight="bold")
fig.savefig(REPO / "figures" / "toc_graphic.png", dpi=600)
fig.savefig(REPO / "figures" / "toc_graphic.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
print("saved", vals)
