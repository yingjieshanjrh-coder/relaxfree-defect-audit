"""Render the figures from the result tables."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from relaxfree.dataset import CLASS_ORDER, FAMILY_ORDER, load_neutral
from relaxfree.plotting import (
    BLUES, CLASS_LABELS, FAMILY_COLORS, GRID, INK, INK_SOFT, PROXY_COLORS, PROXY_LABELS,
    REGIME_LABELS, panel_label, use_style,
)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
TAU = 3.0
ALPHA = 0.1
PROXIES = ["unrelaxed", "identity", "hybrid"]
REGIMES = ["random", "host", "family"]


def save(fig, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png")
    fig.savefig(FIG / f"{name}.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def fig_relaxation(df: pd.DataFrame) -> None:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.6, 3.3), gridspec_kw={"width_ratios": [1.0, 1.35]})

    for fam in FAMILY_ORDER:
        sub = df[df["family"] == fam]
        ax0.scatter(sub["e_relaxed"], sub["e_unrelaxed"], s=7, color=FAMILY_COLORS[fam], alpha=0.55,
                    linewidths=0, label=f"{fam} (n={len(sub)})", rasterized=True)
    lim = (-4, 15)
    ax0.plot(lim, lim, color=INK_SOFT, lw=0.8)
    ax0.axvline(TAU, color=INK_SOFT, lw=0.7, ls=(0, (3, 2)))
    ax0.axhline(TAU, color=INK_SOFT, lw=0.7, ls=(0, (3, 2)))
    ax0.text(-3.6, 14.4, "hits missed by the\nunrelaxed screen", ha="left", va="top", fontsize=7.5, color=INK_SOFT)
    ax0.set(xlim=lim, ylim=lim, xlabel="relaxed formation energy (eV)", ylabel="unrelaxed formation energy (eV)")
    ax0.set_aspect("equal")
    ax0.legend(loc="lower right", handletextpad=0.2, markerscale=1.8)
    panel_label(ax0, "a")

    width = 0.24
    for j, fam in enumerate(FAMILY_ORDER):
        data = [df[(df["family"] == fam) & (df["defect_class"] == c)]["e_relax"].to_numpy() for c in CLASS_ORDER]
        pos = np.arange(len(CLASS_ORDER)) + (j - 1) * (width + 0.03)
        bp = ax1.boxplot(data, positions=pos, widths=width, patch_artist=True, showfliers=False, whis=(10, 90),
                         medianprops={"color": "white", "lw": 1.4}, whiskerprops={"color": FAMILY_COLORS[fam], "lw": 1},
                         capprops={"visible": False}, boxprops={"facecolor": FAMILY_COLORS[fam], "edgecolor": "none"})
        bp["boxes"][0].set_label(fam)
        for x, d in zip(pos, data):
            ax1.text(x, -0.42, str(len(d)), ha="center", va="top", fontsize=6.5, color=INK_SOFT)
    ax1.set_xticks(np.arange(len(CLASS_ORDER)), [CLASS_LABELS[c] for c in CLASS_ORDER], fontsize=7)
    ax1.tick_params(axis="x", pad=11, length=0)
    ax1.set(ylim=(-0.3, 5.2), ylabel="relaxation energy (eV)")
    ax1.grid(axis="x", visible=False)
    ax1.legend(loc="upper left", ncols=3, columnspacing=1.0, handlelength=1.0)
    panel_label(ax1, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig1_relaxation_energy")


def _matrix(ax, values: pd.DataFrame, annot: pd.DataFrame, vmax: float, title: str) -> None:
    im = ax.imshow(values.to_numpy(dtype=float), cmap=BLUES, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(CLASS_ORDER)), [CLASS_LABELS[c] for c in CLASS_ORDER], fontsize=7)
    ax.set_yticks(range(len(FAMILY_ORDER)), FAMILY_ORDER)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(CLASS_ORDER)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(FAMILY_ORDER)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values.iat[i, j]
            dark = np.isfinite(v) and v > 0.55 * vmax
            ax.text(j, i, annot.iat[i, j], ha="center", va="center", fontsize=7,
                    color="white" if dark else INK, linespacing=1.25)
    ax.set_title(title, loc="left", color=INK)
    return im


def _cell_table(tab: pd.DataFrame, col: str) -> pd.DataFrame:
    return tab.pivot(index="family", columns="defect_class", values=col).reindex(index=FAMILY_ORDER, columns=CLASS_ORDER)


def fig_naive_matrix() -> None:
    naive = pd.read_csv(RES / "naive_unrelaxed_screen.csv")
    naive = naive[(naive["tau"] == TAU) & (naive["family"] != "all")]
    rate = _cell_table(naive, "miss_rate")
    hit, missed = _cell_table(naive, "n_hit"), _cell_table(naive, "n_missed")
    annot = pd.DataFrame(
        [[f"{rate.iat[i, j]:.0%}\n{int(missed.iat[i, j])}/{int(hit.iat[i, j])}" for j in range(rate.shape[1])]
         for i in range(rate.shape[0])], index=rate.index, columns=rate.columns)
    fig, ax = plt.subplots(figsize=(5.4, 2.4))
    im = _matrix(ax, rate, annot, 0.7, f"Hits missed when screening on the unrelaxed energy (threshold {TAU:g} eV)")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("miss rate", color=INK)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    save(fig, "fig2_naive_miss_matrix")


def fig_accuracy() -> None:
    acc = pd.read_csv(RES / "proxy_accuracy_overall.csv")
    pick = acc[((acc["proxy"] == "unrelaxed") | (acc["model"] == "hgb")) & acc["proxy"].isin(PROXIES)]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    width = 0.25
    for j, proxy in enumerate(PROXIES):
        vals = [pick[(pick["proxy"] == proxy) & (pick["regime"] == r)]["mae"].iloc[0] for r in REGIMES]
        x = np.arange(len(REGIMES)) + (j - 1) * (width + 0.02)
        ax.bar(x, vals, width, color=PROXY_COLORS[proxy], label=PROXY_LABELS[proxy])
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(REGIMES)), [REGIME_LABELS[r] for r in REGIMES])
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    ax.set(ylabel="MAE of relaxed formation energy (eV)", ylim=(0, 1.6))
    ax.legend(loc="upper left")
    save(fig, "fig3_proxy_accuracy")


def fig_calibration() -> None:
    tab = pd.read_csv(RES / "screen_overall.csv")
    tab = tab[(tab["tau"] == TAU) & (tab["calibration"] == "pooled")]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True, sharey="row")
    for k, regime in enumerate(REGIMES):
        top, bot = axes[0, k], axes[1, k]
        top.plot([0, 0.22], [0, 0.22], color=INK_SOFT, lw=0.8, ls=(0, (3, 2)))
        for proxy in PROXIES:
            sub = tab[(tab["regime"] == regime) & (tab["proxy"] == proxy)].sort_values("alpha")
            top.plot(sub["alpha"], sub["miss_rate"], marker="o", color=PROXY_COLORS[proxy], label=PROXY_LABELS[proxy],
                     markeredgecolor="white", markeredgewidth=0.8)
            bot.plot(sub["alpha"], sub["steps_saved_frac"], marker="o", color=PROXY_COLORS[proxy],
                     markeredgecolor="white", markeredgewidth=0.8)
        top.set_title(REGIME_LABELS[regime], color=INK)
        bot.set_xlabel("target miss rate α")
        bot.set_xticks([0.05, 0.1, 0.2])
        top.set(xlim=(0.03, 0.22), ylim=(0, 0.3))
        bot.set(ylim=(0, 0.7))
    axes[0, 0].set_ylabel("achieved miss rate")
    axes[1, 0].set_ylabel("relaxation steps avoided")
    axes[0, 0].text(0.21, 0.225, "target", ha="right", va="bottom", fontsize=7.5, color=INK_SOFT, rotation=33)
    panel_label(axes[0, 0], "a")
    panel_label(axes[1, 0], "b")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "fig4_calibrated_screening")


CALIBRATION_LABELS = {"pooled": "one margin for all defects", "classwise": "one margin per defect class",
                      "cellwise": "one margin per family and class"}


def fig_validity_matrix(proxy: str, regime: str, name: str) -> None:
    tab = pd.read_csv(RES / "screen_by_cell.csv")
    tab = tab[(tab["tau"] == TAU) & (tab["alpha"] == ALPHA) & (tab["proxy"] == proxy) & (tab["regime"] == regime)]
    fig, axes = plt.subplots(3, 2, figsize=(7.4, 5.6))
    for i, calib in enumerate(CALIBRATION_LABELS):
        sub = tab[tab["calibration"] == calib]
        miss, disc = _cell_table(sub, "miss_rate"), _cell_table(sub, "discard_frac")
        cal = _cell_table(sub, "calibrated_frac")
        a_miss = miss.copy().astype(object)
        a_disc = disc.copy().astype(object)
        for r in range(miss.shape[0]):
            for c in range(miss.shape[1]):
                tag = "" if cal.iat[r, c] > 0.5 else "\nabstain"
                a_miss.iat[r, c] = f"{miss.iat[r, c]:.0%}" + tag
                a_disc.iat[r, c] = f"{disc.iat[r, c]:.0%}" + tag
        _matrix(axes[i, 0], miss, a_miss, 0.4, f"{CALIBRATION_LABELS[calib]}: hits missed")
        _matrix(axes[i, 1], disc, a_disc, 0.8, "candidates discarded")
        if i < 2:
            axes[i, 0].set_xticklabels([])
            axes[i, 1].set_xticklabels([])
        axes[i, 1].set_yticklabels([])
    fig.tight_layout(h_pad=1.0, w_pad=1.2)
    save(fig, name)


def fig_host_transfer() -> None:
    tab = pd.read_csv(RES / "screen_by_host.csv")
    tab = tab[(tab["tau"] == TAU) & (tab["alpha"] == ALPHA) & (tab["calibration"] == "pooled")]
    order = (tab[tab["proxy"] == "unrelaxed"].assign(fam=lambda d: d["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)}))
             .sort_values(["fam", "host"])["host"].tolist())
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    for j, proxy in enumerate(PROXIES):
        sub = tab[tab["proxy"] == proxy].set_index("host").reindex(order)
        x = np.arange(len(order)) + (j - 1) * 0.22
        ax.scatter(x, sub["miss_rate"], s=np.clip(sub["n_hit"] / 5 * 2.0, 6, 60), color=PROXY_COLORS[proxy],
                   edgecolors="white", linewidths=0.5, label=PROXY_LABELS[proxy], zorder=3)
    ax.axhline(ALPHA, color=INK_SOFT, lw=0.8, ls=(0, (3, 2)))
    ax.text(-0.5, ALPHA + 0.01, f"target α = {ALPHA}", ha="left", va="bottom", fontsize=7.5, color=INK_SOFT)
    fams = tab.drop_duplicates("host").set_index("host")["family"].reindex(order)
    start = 0
    for fam in FAMILY_ORDER:
        n = int((fams == fam).sum())
        ax.text(start + (n - 1) / 2, 1.02, fam, transform=ax.get_xaxis_transform(), ha="center", va="bottom", color=INK)
        if start:
            ax.axvline(start - 0.5, color=GRID, lw=1)
        start += n
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=7)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    ax.set(ylabel="miss rate on the unseen host", ylim=(-0.03, 0.8), xlim=(-0.7, len(order) - 0.3))
    ax.legend(loc="upper center", ncols=3, bbox_to_anchor=(0.5, -0.24))
    save(fig, "fig6_host_transfer")


def main() -> None:
    use_style()
    df = load_neutral(ROOT / "data" / "processed" / "defects_neutral.csv")
    fig_relaxation(df)
    fig_naive_matrix()
    fig_accuracy()
    fig_calibration()
    fig_validity_matrix("unrelaxed", "random", "fig5_validity_matrix")
    for proxy in PROXIES:
        for regime in ("random", "host"):
            fig_validity_matrix(proxy, regime, f"si_validity_matrix_{proxy}_{regime}")
    fig_host_transfer()
    print("\n".join(sorted(p.name for p in FIG.glob("*.png"))))


if __name__ == "__main__":
    main()


def fig_external() -> None:
    """Relaxation and reconstruction energies of the external chalcogenide vacancies."""
    use_style()
    main_df = load_neutral(ROOT / "data" / "processed" / "defects_neutral.csv")
    ext = pd.read_csv(ROOT / "data" / "processed" / "external_chalcogenide_vacancies.csv")
    ext = ext[ext["clean"] & (ext["charge"] == 0)]
    check = pd.read_csv(RES / "external_margin_check.csv")
    pooled = check.loc[check["reference"] == "pooled margin", "value_eV"].iloc[0]
    vac = main_df[main_df["defect_class"] == "vacancy"]
    ext_color = "#eda100"

    def ecdf(ax, values, color, label, ls="-"):
        v = np.sort(np.asarray(values, dtype=float))
        ax.step(np.r_[0, v], np.r_[0, np.arange(1, v.size + 1) / v.size], where="post", color=color, lw=2, ls=ls,
                label=label)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for fam in FAMILY_ORDER:
        sub = vac[vac["family"] == fam]["e_relax"]
        ecdf(ax0, sub, FAMILY_COLORS[fam], f"{fam} vacancies, this dataset (n={len(sub)})")
    ecdf(ax0, ext["e_relax"], ext_color, f"chalcogenide cation vacancies, HSE06 (n={len(ext)})")
    ax0.axvline(pooled, color=INK_SOFT, lw=0.8, ls=(0, (3, 2)))
    ax0.text(pooled + 0.05, 0.06, f"pooled margin\n{pooled:.2f} eV", fontsize=7.5, color=INK_SOFT, va="bottom")
    ax0.set(xlim=(0, 3.5), ylim=(0, 1.02), xlabel="relaxation energy from the ideal site (eV)",
            ylabel="cumulative fraction")
    panel_label(ax0, "a")

    rec = ext["e_reconstruction"].dropna()
    ecdf(ax1, rec, ext_color, "lowest distorted start vs ideal-site relaxation")
    ax1.set(xlim=(0, 3.5), ylim=(0, 1.02), xlabel="energy missed by relaxing from the ideal site (eV)",
            ylabel="cumulative fraction")
    frac = (rec > 1.0).mean()
    ax1.text(1.05, 0.45, f"{frac:.0%} of vacancies have a\ndistorted minimum lower\nby more than 1 eV", fontsize=7.5, color=INK, va="top")
    ax1.axvline(1.0, color=GRID, lw=1)
    panel_label(ax1, "b")
    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=2, fontsize=7.5, handlelength=1.4, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(w_pad=2.0, rect=(0, 0.06, 1, 1))
    save(fig, "fig7_external_vacancies")


if __name__ == "__main__" and Path(RES / "external_margin_check.csv").exists():
    fig_external()
