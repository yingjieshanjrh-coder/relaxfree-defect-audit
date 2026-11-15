"""Regression accuracy of the relaxation-free proxies under three transfer regimes."""
from pathlib import Path

import numpy as np
import pandas as pd

from relaxfree.dataset import load_neutral
from relaxfree.features import identity_features
from relaxfree.models import MODEL_NAMES
from relaxfree.proxies import fit_predict
from relaxfree.splits import make_splits

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
N_REPEATS = {"random": 10, "host": 1, "family": 1}
SEED = 11


def main() -> None:
    df = load_neutral(ROOT / "data" / "processed" / "defects_neutral.csv")
    feats = identity_features(df)
    y = df["e_relaxed"].to_numpy()
    records = []
    for regime, n_rep in N_REPEATS.items():
        for i, (train, cal, test) in enumerate(make_splits(regime, df, n_rep, SEED)):
            fit_idx = np.concatenate([train, cal])
            base = {"regime": regime, "split": i, "row": test}
            records.append(pd.DataFrame({**base, "proxy": "unrelaxed", "model": "none",
                                         "pred": df["e_unrelaxed"].to_numpy()[test]}))
            # reference: mean relaxed energy of the training defects of the same class
            class_mean = df.iloc[fit_idx].groupby("defect_class")["e_relaxed"].mean()
            records.append(pd.DataFrame({**base, "proxy": "class_mean", "model": "none",
                                         "pred": df["defect_class"].map(class_mean).to_numpy()[test]}))
            for proxy in ("identity", "hybrid"):
                for model in MODEL_NAMES:
                    pred = fit_predict(proxy, df, feats, fit_idx, test, model, SEED + i)
                    records.append(pd.DataFrame({**base, "proxy": proxy, "model": model, "pred": pred}))
    pred = pd.concat(records, ignore_index=True)
    pred["err"] = pred["pred"] - y[pred["row"].to_numpy()]
    pred["abs_err"] = pred["err"].abs()
    for col in ("family", "defect_class", "interstitial", "host"):
        pred[col] = df[col].to_numpy()[pred["row"].to_numpy()]

    keys = ["regime", "proxy", "model"]
    overall = pred.groupby(keys).agg(mae=("abs_err", "mean"), bias=("err", "mean"),
                                     rmse=("err", lambda e: float(np.sqrt(np.mean(e**2)))), n=("err", "size")).reset_index()
    overall.to_csv(RES / "proxy_accuracy_overall.csv", index=False, float_format="%.4f")
    by_group = pred.groupby(keys + ["interstitial"]).agg(mae=("abs_err", "mean"), n=("err", "size")).reset_index()
    by_group.to_csv(RES / "proxy_accuracy_by_group.csv", index=False, float_format="%.4f")
    by_cell = pred.groupby(keys + ["family", "defect_class"]).agg(mae=("abs_err", "mean"), n=("err", "size")).reset_index()
    by_cell.to_csv(RES / "proxy_accuracy_by_cell.csv", index=False, float_format="%.4f")
    host = pred[pred["regime"] == "host"].groupby(["proxy", "model", "host", "family"]).agg(
        mae=("abs_err", "mean"), n=("err", "size")).reset_index()
    host.to_csv(RES / "proxy_accuracy_by_host.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 200)
    print(overall.pivot_table(index=["proxy", "model"], columns="regime", values="mae").round(3))
    print(by_group.pivot_table(index=["proxy", "model", "interstitial"], columns="regime", values="mae").round(3))


if __name__ == "__main__":
    main()
