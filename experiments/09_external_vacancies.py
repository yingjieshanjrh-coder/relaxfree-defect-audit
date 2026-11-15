"""External check on chalcogenide cation vacancies (HSE06, Mosquera-Lois et al.).

The external relaxations start from the ideal, high-symmetry vacancy
("Unperturbed") and were complemented by bond-distorted starts. The archive has
no host or chemical-potential references, so formation energies and therefore
screening decisions cannot be evaluated directly. What can be tested is the
quantity the decisions depend on: a hit is missed by the screen
`E_unrel - m <= tau` only if its relaxation energy exceeds the margin m, so the
fraction of defects that relax by more than m bounds the miss rate from above.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from relaxfree.conformal import grouped_recall_margin, recall_margin
from relaxfree.dataset import FAMILY_ORDER, load_neutral

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TAU, ALPHA = 3.0, 0.1


def describe(values: np.ndarray, label: str) -> dict:
    v = np.asarray(values, dtype=float)
    return {"set": label, "n": int(v.size), "median": np.median(v), "q75": np.quantile(v, 0.75),
            "q90": np.quantile(v, 0.9), "frac_gt_0p5": np.mean(v > 0.5), "frac_gt_1": np.mean(v > 1.0), "max": v.max()}


def main() -> None:
    main_df = load_neutral(ROOT / "data" / "processed" / "defects_neutral.csv")
    ext = pd.read_csv(ROOT / "data" / "processed" / "external_chalcogenide_vacancies.csv")
    ext_clean = ext[ext["clean"] & (ext["charge"] == 0)]
    ext_usable = ext[ext["usable"] & (ext["charge"] == 0)]
    vac = main_df[main_df["defect_class"] == "vacancy"]

    rows = [describe(vac["e_relax"], "main: all vacancies")]
    rows += [describe(vac.loc[vac["family"] == f, "e_relax"], f"main: {f} vacancies") for f in FAMILY_ORDER]
    rows.append(describe(main_df["e_relax"], "main: all defects"))
    rows.append(describe(ext_clean["e_relax"], "external: clean"))
    rows.append(describe(ext_usable["e_relax"], "external: usable"))
    rows.append(describe(ext_clean["e_relax"] + ext_clean["e_reconstruction"].fillna(0), "external: clean, incl. reconstruction"))
    summary = pd.DataFrame(rows)
    summary.to_csv(RES / "external_relaxation_summary.csv", index=False, float_format="%.4f")

    # margins calibrated on the full main dataset, as a user of the audit would
    y, p = main_df["e_relaxed"].to_numpy(), main_df["e_unrelaxed"].to_numpy()
    pooled = recall_margin(p, y, TAU, ALPHA)
    by_class = grouped_recall_margin(p, y, main_df["defect_class"].to_numpy(), TAU, ALPHA)
    q90_vac = float(np.quantile(vac["e_relax"], 0.9))
    checks = []
    # the class-wise margin for vacancies is negative (their hits sit below the threshold already),
    # so it gives no useful bound and is reported only in the printout
    print(f"vacancy-class margin on the main data: {by_class.get('vacancy', np.inf):.3f} eV")
    for name, m in [("pooled margin", pooled), ("90th percentile of main vacancy relaxation", q90_vac)]:
        for label, sub in [("clean", ext_clean), ("usable", ext_usable)]:
            d = sub["e_relax"].to_numpy()
            dt = d + sub["e_reconstruction"].fillna(0).to_numpy()
            checks.append({"reference": name, "value_eV": m, "external_set": label, "n": len(d),
                           "frac_relax_above": float(np.mean(d > m)),
                           "frac_relax_incl_reconstruction_above": float(np.mean(dt > m)),
                           "frac_main_vacancies_above": float(np.mean(vac["e_relax"] > m))})
    checks = pd.DataFrame(checks)
    checks.to_csv(RES / "external_margin_check.csv", index=False, float_format="%.4f")

    tests = []
    for label, ref in [("main vacancies", vac["e_relax"]), ("main all defects", main_df["e_relax"])]:
        u = mannwhitneyu(ext_clean["e_relax"], ref, alternative="two-sided")
        tests.append({"external": "clean", "reference": label, "U": u.statistic, "p_value": u.pvalue})
    pd.DataFrame(tests).to_csv(RES / "external_tests.csv", index=False, float_format="%.4g")

    rec = ext_clean["e_reconstruction"].dropna()
    recon = pd.DataFrame([{"n": len(rec), "median": rec.median(), "frac_gt_0p1": (rec > 0.1).mean(),
                           "frac_gt_0p5": (rec > 0.5).mean(), "frac_gt_1": (rec > 1.0).mean(), "max": rec.max()}])
    recon.to_csv(RES / "external_reconstruction.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))
    print(checks.round(3).to_string(index=False))
    print(pd.DataFrame(tests).to_string(index=False))
    print(recon.round(3).to_string(index=False))
    print(f"external vacancies: {len(ext)} total, {ext['usable'].sum()} usable, {ext['clean'].sum()} clean; "
          f"neutral clean {len(ext_clean)} in {ext_clean['host'].nunique()} hosts")


if __name__ == "__main__":
    main()
