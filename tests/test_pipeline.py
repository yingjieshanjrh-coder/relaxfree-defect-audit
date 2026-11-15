from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from relaxfree.audit import AuditConfig, run_audit, summarise
from relaxfree.dataset import load_neutral
from relaxfree.features import identity_features
from relaxfree.screening import screening_metrics, wilson_interval
from relaxfree.splits import make_splits

DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "defects_neutral.csv"


@pytest.fixture(scope="module")
def defects() -> pd.DataFrame:
    return load_neutral(DATA)


def test_table_integrity(defects):
    assert defects["defect_id"].is_unique
    assert defects["host"].nunique() == 32
    assert defects["ideal_start"].all()
    assert (defects["start_max_disp"] <= 0.02).all()
    assert (defects["e_relax"] >= -0.1).all()
    assert np.allclose(defects["e_relax"], defects["e_unrelaxed"] - defects["e_relaxed"], atol=1e-5)
    assert (defects["e_unrelaxed"] <= 15.0).all()
    vac = defects[defects["defect_class"] == "vacancy"]
    assert (vac["element"] == "").all() and (vac["replaced"] != "").all()
    inter = defects[defects["interstitial"]]
    assert (inter["replaced"] == "").all() and inter["site"].str.startswith("i_").all()


def test_displaced_starts_are_excluded_by_default(defects):
    everything = load_neutral(DATA, ideal_only=False)
    assert len(everything) > len(defects)
    assert set(defects["defect_id"]) == set(everything.loc[everything["ideal_start"], "defect_id"])


def test_sodium_is_not_read_as_missing(defects):
    assert (defects["element"] == "Na").any()


def test_features_use_no_energies(defects):
    feats = identity_features(defects)
    assert len(feats) == len(defects)
    assert not any(c.startswith("e_") for c in feats.columns)
    shuffled = defects.assign(e_unrelaxed=0.0, e_relaxed=0.0, e_relax=0.0, n_frames=1)
    pd.testing.assert_frame_equal(feats, identity_features(shuffled))


@pytest.mark.parametrize("regime", ["random", "host", "family"])
def test_splits_are_disjoint(defects, regime):
    train, cal, test = next(iter(make_splits(regime, defects, 1, 0)))
    assert not (set(train) & set(cal)) and not (set(train) & set(test)) and not (set(cal) & set(test))
    if regime != "random":
        hosts = defects["host"].to_numpy()
        assert not (set(hosts[test]) & set(hosts[train]))
        assert not (set(hosts[test]) & set(hosts[cal]))
        assert not (set(hosts[cal]) & set(hosts[train]))
    if regime == "family":
        assert defects["family"].to_numpy()[test].tolist().count(defects["family"].to_numpy()[test][0]) == len(test)


def test_screening_metrics():
    y = np.array([1.0, 2.0, 2.5, 4.0, 5.0, 6.0])
    keep = np.array([True, True, False, True, False, False])
    m = screening_metrics(y, keep, tau=3.0, cost=np.array([5, 5, 5, 5, 10, 10]))
    assert m["n_hit"] == 3 and m["n_missed"] == 1
    assert m["miss_rate"] == pytest.approx(1 / 3)
    assert m["discard_frac"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["steps_saved_frac"] == pytest.approx(25 / 40)
    lo, hi = wilson_interval(1, 3)
    assert 0 < lo < 1 / 3 < hi < 1


def test_unrelaxed_audit_runs(defects):
    feats = identity_features(defects)
    cfg = AuditConfig(regimes=["random"], proxies=["unrelaxed"], taus=[3.0], alphas=[0.1], n_repeats={"random": 20})
    dec = run_audit(defects, feats, cfg)
    out = summarise(dec, defects.assign(all="all"), ["all"])
    naive = out[out["calibration"] == "none"].iloc[0]
    pooled = out[out["calibration"] == "pooled"].iloc[0]
    assert naive["miss_rate"] > 0.2
    assert pooled["miss_rate"] < 0.13
    assert pooled["discard_frac"] < naive["discard_frac"]
