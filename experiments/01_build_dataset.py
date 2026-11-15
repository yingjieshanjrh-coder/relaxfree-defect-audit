"""Build the processed defect tables from the upstream label files."""
from pathlib import Path

from relaxfree.dataset import CLASS_ORDER, FAMILY_ORDER, build_tables

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    tables = build_tables(ROOT / "data" / "raw" / "labels", ROOT / "data" / "raw" / "defect_GNN_gen_1")
    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    tables["neutral"].to_csv(out / "defects_neutral.csv", index=False, float_format="%.6f")
    tables["charged"].to_csv(out / "defects_charged.csv", index=False, float_format="%.6f")
    cens_cols = ["host", "family", "defect_class", "site", "element", "replaced", "first_step", "e_unrelaxed", "e_relaxed"]
    tables["censored"][cens_cols].to_csv(out / "defects_censored.csv", index=False, float_format="%.6f")
    tables["single_frame"][cens_cols].to_csv(out / "defects_single_frame.csv", index=False, float_format="%.6f")
    tables["qc"].to_csv(ROOT / "results" / "qc_summary.csv", index=False)

    neutral = tables["neutral"]
    ideal = neutral[neutral["ideal_start"]]
    counts = (
        ideal.groupby(["family", "defect_class"]).size().unstack(fill_value=0)
        .reindex(index=FAMILY_ORDER, columns=CLASS_ORDER)
    )
    start = neutral.groupby(["family", "ideal_start"]).size().unstack(fill_value=0)
    start.to_csv(ROOT / "results" / "start_geometry_by_family.csv")
    print(tables["qc"].to_string(index=False))
    print(f"\nneutral defects kept: {len(neutral)} in {neutral['host'].nunique()} hosts")
    print(start.to_string())
    print(f"\nideal-start defects (main analysis): {len(ideal)} in {ideal['host'].nunique()} hosts")
    print(counts.to_string())
    print(f"charged-state trajectories kept: {len(tables['charged'])}")
    print(f"censored neutral trajectories: {len(tables['censored'])}")


if __name__ == "__main__":
    main()
