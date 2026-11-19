"""Robustness checks: partial relaxation, unseen impurity species, thresholds, costs, models, data choices."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from relaxfree.audit import AuditConfig, add_wilson, run_audit, summarise
from relaxfree.dataset import CLASS_ORDER, FAMILY_ORDER, attach_partial_steps, load_neutral, load_table
from relaxfree.features import identity_features
from relaxfree.parse import PARTIAL_STEPS
from relaxfree.screening import wilson_interval

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PROC = ROOT / "data" / "processed"
TAU, ALPHA = 3.0, 0.1


def save(table: pd.DataFrame, name: str) -> None:
    table.to_csv(RES / name, index=False, float_format="%.4f")


def partial_relaxation(df: pd.DataFrame, feats: pd.DataFrame) -> None:
    """Formation energy after k ionic steps as the screening proxy."""
    proxies = ["unrelaxed"] + [f"step{k}" for k in PARTIAL_STEPS]
    cfg = AuditConfig(proxies=proxies, taus=[TAU], calibrations=["none", "pooled", "merged"])
    dec = run_audit(df, feats, cfg)
    save(add_wilson(summarise(dec, df, ["all"]).drop(columns="all")), "partial_relaxation_screen.csv")
    cell = summarise(dec[dec["alpha"].isna() | (dec["alpha"] == ALPHA)], df, ["family", "defect_class"])
    save(add_wilson(cell), "partial_relaxation_by_cell.csv")
    rows = []
    for p in proxies:
        gap = df["e_unrelaxed" if p == "unrelaxed" else f"e_{p}"] - df["e_relaxed"]
        rows.append({"proxy": p, "median_gap": gap.median(), "q90_gap": gap.quantile(0.9),
                     "frac_gap_gt_1": (gap > 1).mean(), "frac_gap_negative": (gap < -1e-6).mean(),
                     "frac_finished": (df["n_frames"] - 1 <= (0 if p == "unrelaxed" else int(p[4:]))).mean()})
    save(pd.DataFrame(rows), "partial_relaxation_gap.csv")


def unseen_element(df: pd.DataFrame, feats: pd.DataFrame) -> None:
    cfg = AuditConfig(regimes=["element"], taus=[TAU], calibrations=["none", "pooled"])
    dec = run_audit(df, feats, cfg)
    save(add_wilson(summarise(dec, df, ["all"]).drop(columns="all")), "screen_unseen_element.csv")
    by_el = summarise(dec[dec["alpha"].isna() | (dec["alpha"] == ALPHA)], df, ["element"])
    save(add_wilson(by_el), "screen_unseen_element_by_species.csv")


def merge_factor(df: pd.DataFrame, feats: pd.DataFrame) -> None:
    rows = []
    for factor in (1.0, 2.0, 3.0):
        cfg = AuditConfig(proxies=["unrelaxed"], taus=[TAU], alphas=[ALPHA], calibrations=["merged"],
                          merge_factor=factor)
        dec = run_audit(df, feats, cfg)
        overall = summarise(dec, df, ["all"])
        cell = summarise(dec, df, ["family", "defect_class"])
        cell = cell[cell["n_hit"] > 0]
        worst = cell.loc[cell.groupby("regime", observed=True)["miss_rate"].idxmax()].set_index("regime")
        level = dec.groupby(["regime", "level"], observed=True).size().unstack("level", fill_value=0)
        level = level.div(level.sum(axis=1), axis=0)
        for _, r in overall.iterrows():
            reg = r["regime"]
            rows.append({"merge_factor": factor, "min_hits": int(np.ceil(factor * 9)), "regime": reg,
                         "miss_rate": r["miss_rate"], "discard_frac": r["discard_frac"],
                         "steps_saved_frac": r["steps_saved_frac"], "calibrated_frac": r["calibrated_frac"],
                         "worst_cell": f"{worst.loc[reg, 'family']} {worst.loc[reg, 'defect_class']}",
                         "worst_cell_miss": worst.loc[reg, "miss_rate"],
                         **{f"frac_{lv}": level.loc[reg].get(lv, 0.0) for lv in ("cell", "class", "rest")}})
    save(pd.DataFrame(rows), "merged_min_hits_sensitivity.csv")


def single_point_cost(df: pd.DataFrame, feats: pd.DataFrame) -> None:
    """Steps avoided when the ideal-site calculation costs more than one ionic step of the relaxation."""
    cfg = AuditConfig(proxies=["unrelaxed"], taus=[TAU], alphas=[ALPHA], calibrations=["none", "pooled", "merged"])
    dec = run_audit(df, feats, cfg)
    parts = [summarise(dec, df, ["all"], single_point_steps=c).drop(columns="all").assign(single_point_steps=c)
             for c in (1.0, 2.0, 3.0, 5.0)]
    out = pd.concat(parts, ignore_index=True)
    out["mean_steps_per_relaxation"] = df["n_frames"].mean()
    save(out[["single_point_steps", "regime", "calibration", "miss_rate", "discard_frac", "steps_saved_frac",
              "mean_steps_per_relaxation"]], "single_point_cost_sensitivity.csv")


def calibration_size(df: pd.DataFrame) -> None:
    """How many relaxed defects does the pooled margin of the unrelaxed proxy need?"""
    from relaxfree.conformal import keep_mask, recall_margin
    from relaxfree.splits import random_splits

    y, p, steps = df["e_relaxed"].to_numpy(), df["e_unrelaxed"].to_numpy(), df["n_frames"].to_numpy(float)
    rows = []
    for frac in (0.05, 0.1, 0.25, 0.4):
        miss, saved, n_cal_hits, margins = [], [], [], []
        for _, cal, test in random_splits(len(df), 200, 11, frac_train=0.0, frac_cal=frac):
            test = test[: int(0.25 * len(df))]
            m = recall_margin(p[cal], y[cal], TAU, ALPHA)
            keep = keep_mask(p[test], m, TAU)
            hit = y[test] <= TAU
            miss.append((hit & ~keep).sum() / hit.sum())
            saved.append((steps[test][~keep] - 1).sum() / steps[test].sum())
            n_cal_hits.append(int((y[cal] <= TAU).sum()))
            margins.append(m)
        miss, margins = np.array(miss), np.array(margins)
        rows.append({"calibration_frac": frac, "n_calibration": int(frac * len(df)), "mean_calibration_hits": np.mean(n_cal_hits),
                     "mean_miss": miss.mean(), "sd_miss": miss.std(), "q05_miss": np.quantile(miss, 0.05),
                     "q95_miss": np.quantile(miss, 0.95), "frac_repeats_above_15pct": (miss > 0.15).mean(),
                     "mean_steps_saved": np.mean(saved), "frac_no_margin": np.isinf(margins).mean(),
                     "median_margin": np.median(margins[np.isfinite(margins)])})
    save(pd.DataFrame(rows), "calibration_size.csv")


def fixed_offset(df: pd.DataFrame) -> None:
    """Screen on the unrelaxed energy shifted by a fixed offset, as in upper-bound screening with a rule of thumb."""
    y, p = df["e_relaxed"].to_numpy(), df["e_unrelaxed"].to_numpy()
    rows = []
    for tau in [float(t) for t in np.arange(0.0, 6.01, 0.5)]:
        hit = y <= tau
        for off in (0.0, 0.5, 1.0, 1.5):
            keep = p - off <= tau
            missed = int((hit & ~keep).sum())
            lo, hi = wilson_interval(missed, int(hit.sum()))
            rows.append({"tau": tau, "offset": off, "n_hit": int(hit.sum()), "n_missed": missed,
                         "miss_rate": missed / hit.sum() if hit.sum() else np.nan, "miss_lo": lo, "miss_hi": hi,
                         "discard_frac": float((~keep).mean())})
    save(pd.DataFrame(rows), "fixed_offset_screen.csv")
    hit, keep = y <= TAU, p - 1.0 <= TAU
    cell = (df.assign(hit=hit, missed=hit & ~keep).groupby(["family", "defect_class"])[["hit", "missed"]].sum()
            .reset_index())
    cell["miss_rate"] = cell["missed"] / cell["hit"].where(cell["hit"] > 0)
    save(cell, "fixed_offset_by_cell.csv")


def threshold_scan(df: pd.DataFrame, feats: pd.DataFrame) -> None:
    taus = [float(t) for t in np.arange(1.0, 6.01, 0.5)]
    cfg = AuditConfig(regimes=["random"], proxies=["unrelaxed"], taus=taus, alphas=[ALPHA],
                      calibrations=["none", "pooled", "merged"])
    dec = run_audit(df, feats, cfg)
    save(add_wilson(summarise(dec, df, ["all"]).drop(columns="all")), "threshold_scan.csv")


def model_sensitivity(df: pd.DataFrame, feats: pd.DataFrame) -> None:
    parts = []
    for model in ("ridge", "rf"):
        cfg = AuditConfig(proxies=["identity", "hybrid"], taus=[TAU], alphas=[ALPHA], calibrations=["none", "pooled"],
                          model=model, n_repeats={"random": 20, "host": 2, "family": 5})
        dec = run_audit(df, feats, cfg)
        parts.append(add_wilson(summarise(dec, df, ["all"]).drop(columns="all")).assign(model=model))
    save(pd.concat(parts, ignore_index=True), "model_sensitivity.csv")


def chemical_potential(df: pd.DataFrame) -> None:
    """Native defects under the opposite chemical-potential limit (the upstream data give both)."""
    b = pd.read_csv(PROC / "defects_native_b_rich.csv").rename(
        columns={"e_unrelaxed": "e_unrelaxed_b", "e_relaxed": "e_relaxed_b"})
    m = df.merge(b, on="defect_id", how="inner")
    rows = []
    for tau in (2.0, 3.0, 4.0):
        for label, un, rel in (("A-rich", "e_unrelaxed", "e_relaxed"), ("B-rich", "e_unrelaxed_b", "e_relaxed_b")):
            hit = m[rel] <= tau
            missed = hit & (m[un] > tau)
            lo, hi = wilson_interval(int(missed.sum()), int(hit.sum()))
            rows.append({"tau": tau, "condition": label, "n": len(m), "n_hit": int(hit.sum()),
                         "n_missed": int(missed.sum()), "miss_rate": missed.sum() / hit.sum(), "miss_lo": lo,
                         "miss_hi": hi, "median_shift": (m[rel] - m["e_relaxed"]).median(),
                         "max_abs_relax_diff": ((m[un] - m[rel]) - m["e_relax"]).abs().max()})
    save(pd.DataFrame(rows), "chemical_potential_sensitivity.csv")


def selection(df_all: pd.DataFrame) -> None:
    """Do the trajectories excluded for their start geometry differ from the retained ones?"""
    rows = []
    for (fam, ideal), sub in df_all.groupby(["family", "ideal_start"]):
        share = sub["defect_class"].value_counts(normalize=True)
        rows.append({"family": fam, "ideal_start": ideal, "n": len(sub), "n_hosts": sub["host"].nunique(),
                     **{f"share_{c}": share.get(c, 0.0) for c in CLASS_ORDER},
                     "median_e_relaxed": sub["e_relaxed"].median(), "hit_frac_3eV": (sub["e_relaxed"] <= TAU).mean(),
                     "median_e_drop": sub["e_relax"].median()})
    save(pd.DataFrame(rows), "start_geometry_selection.csv")
    # same class composition: compare relaxed energies within II-VI class by class
    ii = df_all[df_all["family"] == "II-VI"]
    rows = []
    for c in CLASS_ORDER:
        a, b = ii[(ii["defect_class"] == c) & ii["ideal_start"]], ii[(ii["defect_class"] == c) & ~ii["ideal_start"]]
        p = mannwhitneyu(a["e_relaxed"], b["e_relaxed"]).pvalue if len(a) and len(b) else np.nan
        rows.append({"defect_class": c, "n_ideal": len(a), "n_displaced": len(b),
                     "median_e_relaxed_ideal": a["e_relaxed"].median(), "median_e_relaxed_displaced": b["e_relaxed"].median(),
                     "hit_frac_ideal": (a["e_relaxed"] <= TAU).mean(), "hit_frac_displaced": (b["e_relaxed"] <= TAU).mean(),
                     "p_value": p})
    save(pd.DataFrame(rows), "start_geometry_selection_iivi.csv")


def bound_and_depth(df: pd.DataFrame) -> None:
    rows = []
    for tau in (2.0, 3.0, 4.0):
        hit = df["e_relaxed"] <= tau
        missed = hit & (df["e_unrelaxed"] > tau)
        depth = tau - df.loc[missed, "e_relaxed"]
        deep_hits = hit & (df["e_relaxed"] <= tau - 1.0)
        rows.append({"tau": tau, "n_missed": int(missed.sum()), "median_depth_missed": depth.median(),
                     "median_depth_hits": (tau - df.loc[hit, "e_relaxed"]).median(),
                     "frac_missed_within_0p5": (depth < 0.5).mean(), "frac_missed_deeper_1": (depth > 1.0).mean(),
                     "miss_rate_deeper_1": (missed & deep_hits).sum() / deep_hits.sum(),
                     "n_negative_relaxation": int((df["e_relax"] < 0).sum()), "min_relaxation": df["e_relax"].min()})
    save(pd.DataFrame(rows), "naive_missed_depth.csv")


def external_host_level(df: pd.DataFrame) -> None:
    """Vacancies in one host are not independent; repeat the comparison on host medians."""
    ext = pd.read_csv(PROC / "external_chalcogenide_vacancies.csv")
    ext = ext[ext["clean"] & (ext["charge"] == 0)]
    vac = df[df["defect_class"] == "vacancy"]
    e_host, v_host = ext.groupby("host")["e_relax"].median(), vac.groupby("host")["e_relax"].median()
    test = mannwhitneyu(e_host, v_host)
    rng = np.random.default_rng(0)
    hosts = e_host.index.to_numpy()
    groups = {h: g["e_relax"].to_numpy() for h, g in ext.groupby("host")}
    boot_med, boot_gt1 = [], []
    for _ in range(5000):
        pick = np.concatenate([groups[h] for h in rng.choice(hosts, hosts.size)])
        boot_med.append(np.median(pick))
        boot_gt1.append((pick > 1).mean())
    save(pd.DataFrame([{
        "n_external_hosts": e_host.size, "n_main_hosts": v_host.size,
        "median_of_host_medians_external": e_host.median(), "median_of_host_medians_main": v_host.median(),
        "U": test.statistic, "p_value": test.pvalue,
        "external_median_lo": np.percentile(boot_med, 2.5), "external_median_hi": np.percentile(boot_med, 97.5),
        "external_gt1_lo": np.percentile(boot_gt1, 2.5), "external_gt1_hi": np.percentile(boot_gt1, 97.5),
    }]), "external_host_level.csv")


def main() -> None:
    df = load_neutral(PROC / "defects_neutral.csv")
    df = attach_partial_steps(df, PROC / "defects_neutral_steps.csv")
    df["all"] = "all"
    feats = identity_features(df)
    bound_and_depth(df)
    chemical_potential(df)
    selection(load_table(PROC / "defects_neutral.csv"))
    external_host_level(df)
    calibration_size(df)
    fixed_offset(df)
    partial_relaxation(df, feats)
    threshold_scan(df, feats)
    single_point_cost(df, feats)
    merge_factor(df, feats)
    unseen_element(df, feats)
    model_sensitivity(df, feats)
    for name in ("naive_missed_depth", "chemical_potential_sensitivity", "external_host_level", "partial_relaxation_gap",
                 "merged_min_hits_sensitivity"):
        print(f"\n{name}")
        print(pd.read_csv(RES / f"{name}.csv").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
