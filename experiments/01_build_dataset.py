"""Build the processed defect tables from the upstream label files."""
from pathlib import Path

from relaxfree.dataset import CLASS_ORDER, FAMILY_ORDER, build_tables

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    tables = build_tables(ROOT / "data" / "raw" / "labels")
    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    tables["neutral"].to_csv(out / "defects_neutral.csv", index=False, float_format="%.6f")
    tables["charged"].to_csv(out / "defects_charged.csv", index=False, float_format="%.6f")
    cens_cols = ["host", "family", "defect_class", "site", "element", "replaced", "first_step", "e_unrelaxed", "e_relaxed"]
    tables["censored"][cens_cols].to_csv(out / "defects_censored.csv", index=False, float_format="%.6f")
    tables["qc"].to_csv(ROOT / "results" / "qc_summary.csv", index=False)

    neutral = tables["neutral"]
    counts = (
        neutral.groupby(["family", "defect_class"]).size().unstack(fill_value=0)
        .reindex(index=FAMILY_ORDER, columns=CLASS_ORDER)
    )
    print(tables["qc"].to_string(index=False))
    print(f"\nneutral defects kept: {len(neutral)} in {neutral['host'].nunique()} hosts")
    print(counts.to_string())
    print(f"charged-state trajectories kept: {len(tables['charged'])}")
    print(f"censored neutral trajectories: {len(tables['censored'])}")


if __name__ == "__main__":
    main()
