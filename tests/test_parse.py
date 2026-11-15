import pandas as pd
import pytest

from relaxfree.parse import host_family, host_species, parse_native_defect, quality_flags


@pytest.mark.parametrize("host, expected", [
    ("GaAs", ("Ga", "As")), ("Si", ("Si", "Si")), ("C", ("C", "C")), ("BN", ("B", "N")),
    ("SnC", ("Sn", "C")), ("CdTe", ("Cd", "Te")), ("BSb", ("B", "Sb")),
])
def test_host_species(host, expected):
    assert host_species(host) == expected


@pytest.mark.parametrize("host, family", [
    ("SiC", "IV"), ("Ge", "IV"), ("GeSn", "IV"), ("InP", "III-V"), ("BAs", "III-V"), ("ZnO", "II-VI"), ("CdS", "II-VI"),
])
def test_host_family(host, family):
    assert host_family(host) == family


def test_native_defect_labels():
    assert parse_native_defect("GaN", "V_N") == {
        "defect_class": "vacancy", "site": "B", "element": None, "replaced": "N"}
    assert parse_native_defect("GaN", "Ga_N") == {
        "defect_class": "antisite", "site": "B", "element": "Ga", "replaced": "N"}
    assert parse_native_defect("InAs", "As_i_neut") == {
        "defect_class": "self_interstitial", "site": "i_neut", "element": "As", "replaced": None}
    assert parse_native_defect("Si", "V_Si")["site"] == "A"
    with pytest.raises(ValueError):
        parse_native_defect("GaN", "V_As")


def test_quality_flags():
    traj = pd.DataFrame({
        "first_step": [0, 2, 0, 0, 0],
        "n_frames": [6, 5, 1, 4, 7],
        "e_unrelaxed": [3.0, 9.0, 2.0, 16.0, 2.0],
        "e_relaxed": [2.5, 4.0, 2.0, 12.0, 2.6],
        "e_min": [2.5, 4.0, 2.0, 12.0, -150.0],
    })
    traj["e_relax"] = traj["e_unrelaxed"] - traj["e_relaxed"]
    flagged = quality_flags(traj)
    assert flagged["keep"].tolist() == [True, False, False, False, False]
    assert flagged["censored"].tolist() == [False, True, False, True, False]
