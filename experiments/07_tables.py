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


def external_tables() -> list[tuple[str, str]]:
    path = RES / "external_relaxation_summary.csv"
    if not path.exists():
        return []
    summ = pd.read_csv(path)
    rows = [[r["set"], str(int(r["n"])), f"{r['median']:.2f}", f"{r['q75']:.2f}", f"{r['q90']:.2f}",
             f"{r['frac_gt_1']:.0%}"] for _, r in summ.iterrows()]
    t1 = md_table(["set", "n", "median dE (eV)", "75th pct", "90th pct", "dE > 1 eV"], rows)
    chk = pd.read_csv(RES / "external_margin_check.csv")
    rows = [[r["reference"], f"{r['value_eV']:.2f}", r["external_set"], str(int(r["n"])),
             f"{r['frac_relax_above']:.0%}", f"{r['frac_relax_incl_reconstruction_above']:.0%}",
             f"{r['frac_main_vacancies_above']:.0%}"] for _, r in chk.iterrows()]
    t2 = md_table(["reference value", "eV", "external set", "n", "external dE above",
                   "external dE incl. reconstruction above", "main vacancies above"], rows)
    return [
        ("External check: relaxation energies of chalcogenide cation vacancies (HSE06) against this dataset", t1),
        ("External check: fraction of vacancies relaxing by more than reference values from this dataset "
         "(an upper bound on the miss rate of hits)", t2),
    ]


def bootstrap_cell_table(calibration: str, regime: str = "random", proxy: str = "unrelaxed") -> str:
    tab = pd.read_csv(RES / "screen_by_cell_bootstrap.csv")
    tab = tab[(tab["calibration"] == calibration) & (tab["regime"] == regime) & (tab["proxy"] == proxy)]

    def fmt(r):
        flag = "**" if r["boot_lo"] > ALPHA else ""
        return f"{flag}{r['miss_rate']:.0%} [{r['boot_lo']:.0%}, {r['boot_hi']:.0%}]{flag} (n = {int(r['n_hit_defects'])})"

    return cell_table(tab, fmt)


def per_repeat_table() -> str:
    tab = pd.read_csv(RES / "screen_per_repeat.csv")
    tab = tab[(tab["tau"] == TAU) & (tab["alpha"] == ALPHA) & (tab["calibration"] == "pooled")]
    rows = []
    for (regime, proxy), sub in tab.groupby(["regime", "proxy"], sort=False):
        m = sub["miss_rate"]
        rows.append([REGIMES[regime], proxy, str(len(m)), f"{m.mean():.1%}", f"{m.std():.1%}",
                     f"{m.quantile(0.05):.1%} - {m.quantile(0.95):.1%}", f"{m.max():.1%}"])
    return md_table(["regime", "proxy", "repetitions", "mean", "s.d.", "5th - 95th pct", "max"], rows)


def partial_table() -> str:
    tab = pd.read_csv(RES / "partial_relaxation_screen.csv")
    tab = tab[tab["regime"] == "random"]
    gap = pd.read_csv(RES / "partial_relaxation_gap.csv").set_index("proxy")
    rows = []
    for proxy in ["unrelaxed", "step1", "step2", "step3", "step5"]:
        sub = tab[tab["proxy"] == proxy]
        raw = sub[sub["calibration"] == "none"].iloc[0]
        row = [proxy, f"{gap.loc[proxy, 'median_gap']:.2f}", f"{gap.loc[proxy, 'q90_gap']:.2f}",
               f"{raw['miss_rate']:.1%} / {raw['steps_saved_frac']:.0%}"]
        for a in (0.05, 0.1, 0.2):
            r = sub[(sub["calibration"] == "pooled") & (sub["alpha"] == a)].iloc[0]
            row.append(f"{r['miss_rate']:.1%} / {r['steps_saved_frac']:.0%}")
        rows.append(row)
    return md_table(["energy used", "median gap to relaxed (eV)", "90th pct gap (eV)", "no margin",
                     "alpha = 0.05", "alpha = 0.1", "alpha = 0.2"], rows)


def element_table() -> str:
    tab = pd.read_csv(RES / "screen_unseen_element.csv")
    rows = []
    for proxy, label in PROXIES.items():
        sub = tab[tab["proxy"] == proxy]
        raw = sub[sub["calibration"] == "none"].iloc[0]
        row = [label, f"{raw['mae']:.2f}", f"{raw['miss_rate']:.1%} / {raw['steps_saved_frac']:.0%}"]
        for a in (0.05, 0.1, 0.2):
            r = sub[(sub["calibration"] == "pooled") & (sub["alpha"] == a)].iloc[0]
            row.append(f"{r['miss_rate']:.1%} [{r['miss_lo']:.1%}, {r['miss_hi']:.1%}] / {r['steps_saved_frac']:.0%}")
        rows.append(row)
    return md_table(["proxy", "MAE (eV)", "no margin", "alpha = 0.05", "alpha = 0.1", "alpha = 0.2"], rows)


def model_table() -> str:
    tab = pd.read_csv(RES / "model_sensitivity.csv")
    main = pd.read_csv(RES / "screen_overall.csv")
    main = main[(main["tau"] == TAU) & (main["calibration"].isin(["none", "pooled"]))
                & (main["alpha"].isna() | (main["alpha"] == ALPHA))].assign(model="hgb")
    tab = pd.concat([tab, main[main["proxy"].isin(["identity", "hybrid"])]], ignore_index=True)
    rows = []
    for proxy in ("identity", "hybrid"):
        for model in ("ridge", "rf", "hgb"):
            row = [proxy, model]
            for regime in REGIMES:
                sub = tab[(tab["proxy"] == proxy) & (tab["model"] == model) & (tab["regime"] == regime)]
                raw, cal = sub[sub["calibration"] == "none"].iloc[0], sub[sub["calibration"] == "pooled"].iloc[0]
                row.append(f"{raw['miss_rate']:.1%} -> {cal['miss_rate']:.1%} / {cal['steps_saved_frac']:.0%}")
            rows.append(row)
    return md_table(["proxy", "model"] + list(REGIMES.values()), rows)


def simple_table(name: str, columns: dict, fmt: dict) -> str:
    tab = pd.read_csv(RES / name)
    rows = [[fmt.get(c, "{}").format(r[c]) for c in columns] for _, r in tab.iterrows()]
    return md_table(list(columns.values()), rows)


def robustness_sections() -> list[tuple[str, str]]:
    pc = "{:.1%}".format
    return [
        (f"Calibrated screens, merged cells. Entries as above", screening_table("merged")),
        ("Unrelaxed proxy, seen hosts, pooled margin: hits missed per cell with 95% intervals from resampling "
         "defects; bold where the lower limit exceeds alpha", bootstrap_cell_table("pooled")),
        ("Unrelaxed proxy, seen hosts, merged cells: hits missed per cell with 95% intervals", bootstrap_cell_table("merged")),
        ("Miss rate of single repetitions, pooled margin", per_repeat_table()),
        ("Partial relaxation as the proxy, seen hosts: hits missed / relaxation steps avoided", partial_table()),
        ("Unseen impurity species (calibration on species unseen by the model): hits missed [95% interval] / "
         "relaxation steps avoided", element_table()),
        ("Regression model behind the learned proxies: hits missed without margin -> with pooled margin / "
         "relaxation steps avoided", model_table()),
        ("Minimum number of hits per block of the merged taxonomy (unrelaxed proxy)", simple_table(
            "merged_min_hits_sensitivity.csv",
            {"min_hits": "min hits", "regime": "regime", "miss_rate": "hits missed", "steps_saved_frac": "steps avoided",
             "calibrated_frac": "calibrated", "frac_cell": "own cell", "frac_class": "merged in class",
             "frac_rest": "rest block", "worst_cell": "worst cell", "worst_cell_miss": "worst cell missed"},
            {c: "{:.1%}" for c in ("miss_rate", "steps_saved_frac", "calibrated_frac", "frac_cell", "frac_class",
                                   "frac_rest", "worst_cell_miss")})),
        ("Native defects under both chemical-potential limits: unrelaxed screen", simple_table(
            "chemical_potential_sensitivity.csv",
            {"tau": "tau (eV)", "condition": "condition", "n": "n", "n_hit": "hits", "n_missed": "missed",
             "miss_rate": "hits missed", "miss_lo": "lower", "miss_hi": "upper"},
            {"tau": "{:g}", "miss_rate": "{:.1%}", "miss_lo": "{:.1%}", "miss_hi": "{:.1%}"})),
        ("How far below the threshold the hits missed by the unrelaxed screen lie", simple_table(
            "naive_missed_depth.csv",
            {"tau": "tau (eV)", "n_missed": "missed", "median_depth_missed": "median depth of missed hits (eV)",
             "median_depth_hits": "median depth of all hits (eV)", "frac_missed_within_0p5": "missed within 0.5 eV",
             "miss_rate_deeper_1": "miss rate of hits deeper than 1 eV"},
            {"tau": "{:g}", "median_depth_missed": "{:.2f}", "median_depth_hits": "{:.2f}",
             "frac_missed_within_0p5": "{:.0%}", "miss_rate_deeper_1": "{:.1%}"})),
        ("External check repeated on host medians", simple_table(
            "external_host_level.csv",
            {"n_external_hosts": "external hosts", "n_main_hosts": "zinc blende hosts",
             "median_of_host_medians_external": "external median (eV)", "median_of_host_medians_main": "zinc blende median (eV)",
             "p_value": "p", "external_median_lo": "external median, lower", "external_median_hi": "upper"},
            {"median_of_host_medians_external": "{:.2f}", "median_of_host_medians_main": "{:.2f}", "p_value": "{:.3f}",
             "external_median_lo": "{:.2f}", "external_median_hi": "{:.2f}"})),
    ]


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
    ] + external_tables() + robustness_sections()
    body = "# Result tables\n\nGenerated by `experiments/07_tables.py` from the CSV files in this directory.\n"
    for i, (title, table) in enumerate(sections, 1):
        body += f"\n## Table {i}. {title}\n\n{table}\n"
    (RES / "tables.md").write_text(body)
    refresh_readme(screening_table("pooled", bold_above=1.5 * ALPHA))
    print(body)


if __name__ == "__main__":
    main()
