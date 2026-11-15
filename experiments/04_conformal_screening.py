"""Calibrated screening: how much relaxation can be skipped at a guaranteed miss rate?"""
from pathlib import Path

import pandas as pd

from relaxfree.audit import AuditConfig, add_wilson, per_repeat_miss, run_audit, summarise
from relaxfree.dataset import load_neutral
from relaxfree.features import identity_features

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def main() -> None:
    df = load_neutral(ROOT / "data" / "processed" / "defects_neutral.csv")
    feats = identity_features(df)
    df["all"] = "all"
    cfg = AuditConfig()
    decisions = run_audit(df, feats, cfg)

    overall = add_wilson(summarise(decisions, df, ["all"]).drop(columns="all"))
    overall.to_csv(RES / "screen_overall.csv", index=False, float_format="%.4f")
    add_wilson(summarise(decisions, df, ["interstitial"])).to_csv(RES / "screen_by_group.csv", index=False, float_format="%.4f")
    add_wilson(summarise(decisions, df, ["family", "defect_class"])).to_csv(RES / "screen_by_cell.csv", index=False, float_format="%.4f")
    add_wilson(summarise(decisions, df, ["family"])).to_csv(RES / "screen_by_family.csv", index=False, float_format="%.4f")
    host = summarise(decisions[decisions["regime"] == "host"], df, ["host", "family"])
    add_wilson(host).to_csv(RES / "screen_by_host.csv", index=False, float_format="%.4f")
    per_repeat_miss(decisions, df).to_csv(RES / "screen_per_repeat.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 220)
    view = overall[(overall["tau"] == 3.0) & (overall["alpha"].isna() | (overall["alpha"] == 0.1))]
    cols = ["regime", "proxy", "calibration", "n_hit", "miss_rate", "discard_frac", "steps_saved_frac", "precision", "calibrated_frac"]
    print(view[cols].sort_values(["regime", "proxy", "calibration"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
