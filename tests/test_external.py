import json

import numpy as np
import pandas as pd

from relaxfree.external import join_segments, segment_order, summarise_segments


def _seg(name, energies, nsc, complete=1, nelm=100):
    return {"file": name, "n_steps": len(energies), "complete": complete, "nelm": nelm,
            "energies": json.dumps(energies), "n_scsteps": json.dumps(nsc)}


def test_segment_order_puts_plain_file_last():
    names = ["vasprun.xml", "vasprun.xml_22_10_53on06_03_23.gz", "vasprun.xml_08_37_02on06_03_23.gz",
             "vasprun.xml_13_56_14on07_03_23.gz"]
    assert sorted(names, key=segment_order) == [names[2], names[1], names[3], names[0]]


def test_join_segments_concatenates_in_time_order():
    seg = pd.DataFrame([
        _seg("vasprun.xml", [-10.30, -10.31], [5, 4]),
        _seg("vasprun.xml_08_00_00on01_01_23.gz", [-9.0, -10.0], [20, 10], complete=0),
        _seg("vasprun.xml_09_00_00on01_01_23.gz", [-10.0, -10.3], [8, 6], complete=0),
    ])
    tr = join_segments(seg)
    assert tr["n_segments"] == 3 and tr["n_steps"] == 6
    assert tr["e_first"] == -9.0 and tr["e_last"] == -10.31
    assert tr["max_join"] < 1e-9 and tr["ends_converged"] and tr["final_complete"]


def test_unconverged_ends_and_jumps_are_flagged():
    seg = pd.DataFrame([
        _seg("vasprun.xml_08_00_00on01_01_23.gz", [-9.0, -10.0], [100, 10], complete=0),
        _seg("vasprun.xml", [-10.5, -10.6], [5, 4]),
    ])
    tr = join_segments(seg)
    assert not tr["ends_converged"]
    assert abs(tr["max_join"] - 0.5) < 1e-9


def test_summary_relaxation_and_reconstruction():
    rows = []
    for dist, energies in [("Unperturbed", [-10.0, -10.4]), ("Bond_Distortion_30.0%", [-9.5, -11.0]),
                           ("Bond_Distortion_0.0%", [-9.9, -10.3])]:
        rows.append({"top": "01_Training", "host": "X2S3", "defect": "v_X_s0_0", "distortion": dist,
                     **_seg("vasprun.xml", energies, [5] * len(energies))})
    out = summarise_segments(pd.DataFrame(rows)).iloc[0]
    assert out["charge"] == 0 and out["element"] == "X"
    assert np.isclose(out["e_relax"], 0.4)
    assert np.isclose(out["e_reconstruction"], 0.6)
    assert out["best_distortion"] == "Bond_Distortion_30.0%"
    assert out["clean"] and out["usable"]
