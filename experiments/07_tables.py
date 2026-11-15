"""Write the summary tables as Markdown from the result files and refresh the README table."""
from pathlib import Path

import pandas as pd

from relaxfree.dataset import CLASS_ORDER, FAMILY_ORDER, load_neutral

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TAU, ALPHA = 3.0, 0.1
PROXIES = {"unrelaxed": "unrelaxed energy", "identity": "identity-only model",
           "hybrid": "unrelaxed + learned relaxation energy"}
REGIMES = {"random": "seen hosts", "host": "unseen host", "family": "unseen family"}
CLASS_HEAD = {"vacancy": "vacancy", "antisite": "antisite", "sub_impurity": "subst. impurity",
              "self_interstitial": "self-interstitial", "int_impurity": "interst. impurity"}


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def cell_table(tab: pd.DataFrame, fmt) -> str:
    rows = []
    for fam in FAMILY_ORDER:
        row = [fam]
        for cls in CLASS_ORDER:
            rec = tab[(tab["family"] == fam) & (tab["defect_class"] == cls)]
            row.append(fmt(rec.iloc[0]) if len(rec) else "-")
        rows.append(row)
    return md_table(["family"] + [CLASS_HEAD[c] for c in CLASS_ORDER], rows)


def counts_table(df: pd.DataFrame) -> str:
    tab = df.groupby(["family", "defect_class"]).size().rename("n").reset_index()
    return cell_table(tab, lambda r: str(int(r["n"])))


def relaxation_table() -> str:
    tab = pd.read_csv(RES / "relaxation_by_cell.csv")
    return cell_table(tab, lambda r: f"{r['median']:.2f} ({r['q25']:.2f}-{r['q75']:.2f})")


def start_table() -> str:
    tab = pd.read_csv(RES / "relaxation_by_start_geometry.csv")
    rows = [[r["family"], "ideal sites" if r["ideal_start"] else "displaced", str(int(r["n"])),
             f"{r['median']:.2f}", f"{r['q75']:.2f}"] for _, r in tab.iterrows()]
    return md_table(["family", "first frame", "n", "median dE (eV)", "75th pct dE (eV)"], rows)


def naive_table() -> str:
    tab = pd.read_csv(RES / "naive_unrelaxed_screen.csv")
    tab = tab[tab["tau"] == TAU]
    return cell_table(tab, lambda r: f"{r['miss_rate']:.0%} ({int(r['n_missed'])}/{int(r['n_hit'])})")


def sensitivity_table() -> str:
    tab = pd.read_csv(RES / "naive_unrelaxed_sensitivity.csv")
    cols = ["main", "with_displaced_starts", "with_censored", "with_single_frame", "e_min_as_relaxed"]
    rows = [[f"{r['tau']:g}"] + [f"{r[c]:.1%}" for c in cols] for _, r in tab.iterrows()]
    return md_table(["tau (eV)", "main analysis", "with displaced-start trajectories", "with censored trajectories", "with single-frame trajectories",
                     "lowest frame as relaxed energy"], rows)


def accuracy_table() -> str:
    tab = pd.read_csv(RES / "proxy_accuracy_overall.csv")
    rows = []
    for proxy in ["class_mean", "unrelaxed", "identity", "hybrid"]:
        for model in ["none", "ridge", "rf", "hgb"]:
            sub = tab[(tab["proxy"] == proxy) & (tab["model"] == model)]
            if sub.empty:
                continue
            mae = sub.set_index("regime")["mae"]
            rows.append([proxy, model] + [f"{mae[r]:.2f}" for r in REGIMES])
    return md_table(["proxy", "model"] + list(REGIMES.values()), rows)


def uncalibrated_table() -> str:
    tab = pd.read_csv(RES / "screen_overall.csv")
    tab = tab[(tab["tau"] == TAU) & (tab["calibration"] == "none")]
    rows = []
    for proxy, label in PROXIES.items():
        row = [label]
        for regime in REGIMES:
            r = tab[(tab["proxy"] == proxy) & (tab["regime"] == regime)].iloc[0]
            row.append(f"{r['miss_rate']:.1%} / {r['discard_frac']:.0%}")
        rows.append(row)
    return md_table(["proxy"] + list(REGIMES.values()), rows)


def screening_table(calibration: str = "pooled", bold_above: float | None = None) -> str:
    tab = pd.read_csv(RES / "screen_overall.csv")
    tab = tab[(tab["tau"] == TAU) & (tab["alpha"] == ALPHA) & (tab["calibration"] == calibration)]
    rows = []
    for proxy, label in PROXIES.items():
        row = [label]
        for regime in REGIMES:
            r = tab[(tab["proxy"] == proxy) & (tab["regime"] == regime)].iloc[0]
            miss = f"{r['miss_rate']:.1%}"
            if bold_above is not None and r["miss_rate"] > bold_above:
                miss = f"**{miss}**"
            row.append(f"{miss} / {r['discard_frac']:.0%} / {r['steps_saved_frac']:.0%} / {1 - r['calibrated_frac']:.0%}")
        rows.append(row)
    return md_table(["proxy"] + list(REGIMES.values()), rows)


def all_operating_points() -> str:
    tab = pd.read_csv(RES / "screen_overall.csv")
    tab = tab[tab["calibration"] == "pooled"]
    rows = []
    for (regime, proxy, tau), sub in tab.groupby(["regime", "proxy", "tau"], sort=False):
        sub = sub.set_index("alpha")
        rows.append([REGIMES[regime], proxy, f"{tau:g}"] + [
            f"{sub.loc[a, 'miss_rate']:.1%} / {sub.loc[a, 'steps_saved_frac']:.0%}" for a in (0.05, 0.1, 0.2)])
    return md_table(["regime", "proxy", "tau (eV)", "alpha = 0.05", "alpha = 0.1", "alpha = 0.2"], rows)


def validity_table(calibration: str, regime: str = "random", proxy: str = "unrelaxed") -> str:
    tab = pd.read_csv(RES / "screen_by_cell.csv")
    tab = tab[(tab["tau"] == TAU) & (tab["alpha"] == ALPHA) & (tab["calibration"] == calibration)
              & (tab["regime"] == regime) & (tab["proxy"] == proxy)]

    def fmt(r):
        if r["calibrated_frac"] < 0.5:
            return f"abstain ({r['calibrated_frac']:.0%} calibrated)"
        return f"{r['miss_rate']:.0%} / {r['discard_frac']:.0%}"

    return cell_table(tab, fmt)


def charged_table() -> str:
    tab = pd.read_csv(RES / "charged_state_screen.csv")
    tab = tab[tab["tau"] == TAU]
    rows = [[f"{int(r['charge']):+d}", "interstitial" if r["interstitial"] else "lattice site", str(int(r["n"])),
             f"{r['median_e_relax']:.2f}", f"{r['q90_e_relax']:.2f}", f"{r['miss_rate']:.1%} ({int(r['n_missed'])}/{int(r['n_hit'])})"]
            for _, r in tab.iterrows()]
    return md_table(["charge", "site type", "n", "median dE (eV)", "90th pct dE (eV)", "hits missed"], rows)


def refresh_readme(table: str) -> None:
    path = ROOT / "README.md"
    text = path.read_text()
    start, end = "<!-- screening-table:start -->", "<!-- screening-table:end -->"
    i, j = text.index(start) + len(start), text.index(end)
    path.write_text(text[:i] + "\n" + table + "\n" + text[j:])


def main() -> None:
    df = load_neutral(ROOT / "data" / "processed" / "defects_neutral.csv")
    legend = "Entries: hits missed / candidates discarded / relaxation steps avoided / candidates in uncalibrated groups."
    sections = [
        ("Number of neutral defects whose first frame is the ideal-site geometry (main analysis)", counts_table(df)),
        ("All kept neutral trajectories by geometry of the first frame", start_table()),
        ("Relaxation energy in eV: median (interquartile range)", relaxation_table()),
        (f"Hits missed by the unrelaxed screen, tau = {TAU:g} eV", naive_table()),
        ("Sensitivity of the unrelaxed-screen miss rate to data choices", sensitivity_table()),
        ("MAE of the relaxed formation energy (eV)", accuracy_table()),
        (f"Uncalibrated screens, tau = {TAU:g} eV: hits missed / candidates discarded", uncalibrated_table()),
        (f"Calibrated screens, pooled margin, tau = {TAU:g} eV, alpha = {ALPHA:g}. {legend}", screening_table("pooled")),
        (f"Calibrated screens, class-wise margins. {legend}", screening_table("classwise")),
        (f"Calibrated screens, cell-wise margins. {legend}", screening_table("cellwise")),
        ("Pooled margin at all operating points: hits missed / relaxation steps avoided", all_operating_points()),
        ("Unrelaxed proxy, seen hosts, pooled margin: hits missed / candidates discarded per cell", validity_table("pooled")),
        ("Unrelaxed proxy, seen hosts, cell-wise margins", validity_table("cellwise")),
        ("Unrelaxed proxy, unseen host, cell-wise margins", validity_table("cellwise", "host")),
        (f"Charged states screened on the single point at the neutral geometry, tau = {TAU:g} eV", charged_table()),
    ]
    body = "# Result tables\n\nGenerated by `experiments/07_tables.py` from the CSV files in this directory.\n"
    for i, (title, table) in enumerate(sections, 1):
        body += f"\n## Table {i}. {title}\n\n{table}\n"
    (RES / "tables.md").write_text(body)
    refresh_readme(screening_table("pooled", bold_above=1.5 * ALPHA))
    print(body)


if __name__ == "__main__":
    main()
