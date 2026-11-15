"""Why the CdSe(x)Te(1-x) alloy subset of the upstream data is not used as a test set.

For each neutral trajectory in Dataset_3 this records how far the first frame is
from the ideal zinc blende sites and how much the structure changes between the
first and the last frame. The screening question needs a first frame that is
the defect on its ideal site in a relaxed host; the table shows that this
subset does not provide one.
"""
import re
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from relaxfree.geometry import IDEAL_TOL, displacement_from_ideal
from relaxfree.structures import displacement_stats

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "raw" / "defect_GNN_gen_1" / "Dataset_3" / "Dataset_3.zip"
PATTERN = re.compile(r"^(\d+)-(CdTe|CdSe|Cd_Te[0-9.]+_Se[0-9.]+)-(.+?)-POSCAR(\d+)-Neutral\.cif$")
LABELS = "Dataset_3/A_rich/Neutral/id_prop_A_rich.csv"


def main() -> None:
    zf = zipfile.ZipFile(ARCHIVE)
    energies = pd.read_csv(zf.open(LABELS), header=None, names=["file", "dfe"]).set_index("file")["dfe"]
    frames: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for name in zf.namelist():
        if "A_rich/Neutral/" not in name or "__MACOSX" in name or not name.endswith(".cif"):
            continue
        m = PATTERN.match(Path(name).name)
        if m:
            frames.setdefault((m[2], m[3]), []).append((int(m[4]), name))

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        first_path, last_path = Path(tmp) / "first.cif", Path(tmp) / "last.cif"
        for (host, defect), fr in sorted(frames.items()):
            fr.sort()
            first, last = fr[0][1], fr[-1][1]
            n_disp, max_disp = displacement_from_ideal(zf.read(first).decode())
            row = {
                "host": host, "defect": defect, "n_frames": len(fr),
                "start_n_displaced": n_disp, "start_max_disp": max_disp,
                "e_first": energies[Path(first).name], "e_last": energies[Path(last).name],
            }
            if len(fr) > 1:
                first_path.write_bytes(zf.read(first))
                last_path.write_bytes(zf.read(last))
                st = displacement_stats(first_path, last_path)
                row.update(frac_moved=st["frac_moved"], u_median=st["u_median"])
            rows.append(row)
    out = pd.DataFrame(rows)
    out["e_change"] = out["e_first"] - out["e_last"]
    out.to_csv(ROOT / "results" / "alloy_subset_check.csv", index=False, float_format="%.4f")

    multi = out[out["n_frames"] > 1]
    print(f"trajectories: {len(out)}, multi-frame: {len(multi)}")
    print(f"first frame on ideal sites (max displacement <= {IDEAL_TOL} A): {(out['start_max_disp'] <= IDEAL_TOL).sum()}")
    print(f"multi-frame with < 5% of atoms moving and |dE| < 0.1 eV: "
          f"{((multi['frac_moved'] < 0.05) & (multi['e_change'].abs() < 0.1)).sum()}")
    print(f"multi-frame with > 50% of atoms moving: {(multi['frac_moved'] > 0.5).sum()}")


if __name__ == "__main__":
    main()
