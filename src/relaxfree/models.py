"""Regressors used as relaxation-free proxies."""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_model(name: str, seed: int = 0):
    if name == "ridge":
        return make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0.0),
            StandardScaler(),
            RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]),
        )
    if name == "rf":
        return make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0.0),
            RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed),
        )
    if name == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=5,
            l2_regularization=1.0, random_state=seed,
        )
    raise ValueError(f"unknown model {name!r}")


MODEL_NAMES = ["ridge", "rf", "hgb"]
