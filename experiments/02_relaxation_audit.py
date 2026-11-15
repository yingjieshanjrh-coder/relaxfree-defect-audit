"""How much does relaxation matter, and where? Model-free audit of the unrelaxed proxy."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from relaxfree.dataset import CLASS_ORDER, FAMILY_ORDER, load_neutral
from relaxfree.screening import metrics_by_cell, screening_metrics

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TAUS = [2.0, 3.0, 4.0]
CELL = ["family", "defect_class"]


def relaxation_summary(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = df.groupby(by)["e_relax"]
    out = g.agg(
        n="size", median="median", mean="mean",
        q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75), q90=lambda s: s.quantile(0.90),
        frac_gt_0p5=lambda s: (s > 0.5).mean(), frac_gt_1=lambda s: (s > 1.0).mean(), max="max",
    )
    return out.reset_index()


def rank_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of unrelaxed vs relaxed energies inside each host."""
    rows = []
    for (host, inter), sub in df.groupby(["host", "interstitial"]):
        if len(sub) < 8:
            continue
        rho = spearmanr(sub["e_unrelaxed"], sub["e_relaxed"]).statistic
        k = max(3, len(sub) // 5)
        top_true = set(sub.nsmallest(k, "e_relaxed")["defect_id"])
        top_proxy = set(sub.nsmallest(k, "e_unrelaxed")["defect_id"])
        rows.append({
            "host": host, "family": sub["family"].iloc[0], "interstitial": inter, "n": len(sub),
            "spearman": rho, "topk": k, "topk_overlap": len(top_true & top_proxy) / k,
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = load_neutral(ROOT / "data" / "processed" / "defects_neutral.csv")
    RES.mkdir(exist_ok=True)

    cells = relaxation_summary(df, CELL)
    cells["family"] = pd.Categorical(cells["family"], FAMILY_ORDER)
    cells["defect_class"] = pd.Categorical(cells["defect_class"], CLASS_ORDER)
    cells = cells.sort_values(CELL)
    cells.to_csv(RES / "relaxation_by_cell.csv", index=False, float_format="%.4f")
    relaxation_summary(df, ["host"]).to_csv(RES / "relaxation_by_host.csv", index=False, float_format="%.4f")
    relaxation_summary(df, ["defect_class", "site"]).to_csv(RES / "relaxation_by_site.csv", index=False, float_format="%.4f")

    naive = []
    for tau in TAUS:
        keep = df["e_unrelaxed"].to_numpy() <= tau
        m = metrics_by_cell(df, keep, tau, CELL)
        m.insert(0, "tau", tau)
        naive.append(m)
        allm = screening_metrics(df["e_relaxed"], keep, tau, df["n_frames"])
        naive.append(pd.DataFrame([{"tau": tau, "family": "all", "defect_class": "all", **allm}]))
    naive = pd.concat(naive, ignore_index=True)
    naive.to_csv(RES / "naive_unrelaxed_screen.csv", index=False, float_format="%.4f")

    ranks = rank_agreement(df)
    ranks.to_csv(RES / "rank_agreement_by_host.csv", index=False, float_format="%.4f")

    # sensitivity of the unrelaxed-screen miss rate to the data choices:
    #  censored      trajectories whose ideal-site frame is missing (> 15 eV) can only be discarded
    #  single_frame  trajectories with one recorded frame, counted as defects that do not relax
    #  e_min         lowest energy along the trajectory instead of the last frame as the relaxed energy
    proc = ROOT / "data" / "processed"
    cens = pd.read_csv(proc / "defects_censored.csv")
    single = pd.read_csv(proc / "defects_single_frame.csv")
    sens = []
    for tau in TAUS:
        hit = df["e_relaxed"] <= tau
        missed = hit & (df["e_unrelaxed"] > tau)
        hit_min = df["e_min"] <= tau
        missed_min = hit_min & (df["e_unrelaxed"] > tau)
        hits_c = int((cens["e_relaxed"] <= tau).sum())
        hits_s = int((single["e_relaxed"] <= tau).sum())
        sens.append({
            "tau": tau,
            "main": missed.sum() / hit.sum(),
            "with_censored": (missed.sum() + hits_c) / (hit.sum() + hits_c),
            "with_single_frame": missed.sum() / (hit.sum() + hits_s),
            "e_min_as_relaxed": missed_min.sum() / hit_min.sum(),
            "n_hits_main": int(hit.sum()), "n_hits_censored": hits_c, "n_hits_single_frame": hits_s,
        })
    pd.DataFrame(sens).to_csv(RES / "naive_unrelaxed_sensitivity.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 200)
    print(cells.to_string(index=False))
    print()
    print(naive[naive["tau"] == 3.0][["family", "defect_class", "n", "n_hit", "n_missed", "miss_rate", "discard_frac"]].to_string(index=False))
    print()
    print(ranks.groupby(["family", "interstitial"])[["spearman", "topk_overlap"]].median().round(3))
    print(pd.DataFrame(sens).round(3).to_string(index=False))
    print("overall e_relax quantiles:", np.round(df["e_relax"].quantile([0.25, 0.5, 0.75, 0.9, 0.99]).to_numpy(), 3))


if __name__ == "__main__":
    main()
