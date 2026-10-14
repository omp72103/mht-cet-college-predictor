"""
Core model code for the MHT-CET college predictor.

Pipeline
--------
1. Point model: five regressors (RandomForest + four HistGradientBoosting variants)
   predict a college/branch/category cutoff percentile. Blend weights come from
   non-negative least squares (NNLS) on an internal validation carve-out.
2. Uncertainty model (SigmaModel): a boosted regressor that predicts the ensemble's
   own absolute error per row, so "how sure are we about this cutoff" varies by row.
3. Calibration: z = (student_percentile - predicted_cutoff) / sigma is mapped to an
   admission probability by isotonic regression fitted on a held-out calibration slice.

Deliberately NOT used as a feature: `sml_rank` (correlation about -0.98 with the target,
i.e. target leakage).

Keep this module importable under a stable name: joblib pickles reference these classes.
Run scripts from the `ml/` directory (``python -m src.train``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.features import allocation_level, eligible_category_prefixes, extract_region, gender_scope

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

CATEGORICAL = ["college_code", "branch", "category", "gender_scope", "allocation_level", "seat_source", "region"]
NUMERIC = ["round"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "percentile"
HGB_CATEGORICAL = [c for c in CATEGORICAL if c != "college_code"]

RANDOM_STATE = 42
OFFSETS = np.arange(-25, 26, 5)
SIGMA_FLOOR = 4.0   # never claim tighter certainty than this many percentile points
SIGMA_CEIL = 30.0   # never claim wider uncertainty than this


# --------------------------------------------------------------------------- data
def load_data(path: Path | str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or DATA_DIR / "combined_cutoffs.csv")
    df["college_code"] = df["college_code"].astype(str)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    df["region"] = df["college_name"].apply(extract_region)
    df["gender_scope"] = df["category"].apply(gender_scope)
    df["allocation_level"] = df["category"].apply(allocation_level)
    return df


# --------------------------------------------------------------------- components
class CollegeTargetEncoder:
    """Smoothed mean-percentile encoding of college_code (366 values exceeds HGB's
    255-category native limit). Fit on training rows only."""

    def __init__(self, smoothing: float = 25.0):
        self.smoothing = smoothing

    def fit(self, codes: pd.Series, target):
        target = pd.Series(np.asarray(target), index=codes.index)
        self.global_mean_ = float(target.mean())
        stats = target.groupby(codes).agg(["mean", "count"])
        self.map_ = (
            (stats["count"] * stats["mean"] + self.smoothing * self.global_mean_)
            / (stats["count"] + self.smoothing)
        ).to_dict()
        return self

    def transform(self, codes: pd.Series) -> np.ndarray:
        return codes.map(self.map_).fillna(self.global_mean_).to_numpy()


def build_rf_pipeline() -> Pipeline:
    # max_features is left at its default: restricting it hurts high-cardinality one-hot columns.
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)], remainder="passthrough")
    rf = RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_leaf=2,
                               random_state=RANDOM_STATE, n_jobs=-1)
    return Pipeline([("pre", pre), ("model", rf)])


class _HGBBase:
    """Shared featurisation for HGB-based models (ordinal cats + target-encoded college)."""

    def __init__(self, target_smoothing: float = 25.0):
        self.cat_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.college_enc = CollegeTargetEncoder(smoothing=target_smoothing)

    def _featurize(self, X: pd.DataFrame) -> np.ndarray:
        cat_part = self.cat_enc.transform(X[HGB_CATEGORICAL])
        college_part = self.college_enc.transform(X["college_code"]).reshape(-1, 1)
        return np.hstack([cat_part, college_part, X[NUMERIC].to_numpy()])


class HGBModel(_HGBBase):
    """Picklable HistGradientBoostingRegressor + its fitted encoders."""

    def __init__(self, random_state=RANDOM_STATE, max_depth=8, learning_rate=0.06, max_iter=500,
                 l2_regularization=0.3, target_smoothing=25.0):
        super().__init__(target_smoothing)
        self.model = HistGradientBoostingRegressor(
            max_iter=max_iter, max_depth=max_depth, learning_rate=learning_rate,
            l2_regularization=l2_regularization,
            categorical_features=list(range(len(HGB_CATEGORICAL))),
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=25,
            random_state=random_state,
        )

    def fit(self, X, y):
        self.college_enc.fit(X["college_code"], y)
        self.cat_enc.fit(X[HGB_CATEGORICAL])
        self.model.fit(self._featurize(X), y)
        return self

    def predict(self, X):
        return self.model.predict(self._featurize(X))


class SigmaModel(_HGBBase):
    """Per-row uncertainty: predicts the point ensemble's |residual|, clipped to [4, 30]."""

    def __init__(self):
        super().__init__(25.0)
        self.model = HistGradientBoostingRegressor(
            max_iter=250, max_depth=5, learning_rate=0.07, l2_regularization=0.5,
            categorical_features=list(range(len(HGB_CATEGORICAL))),
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=15,
            random_state=RANDOM_STATE,
        )

    def fit(self, X, abs_residual):
        self.college_enc.fit(X["college_code"], abs_residual)
        self.cat_enc.fit(X[HGB_CATEGORICAL])
        self.model.fit(self._featurize(X), abs_residual)
        return self

    def predict(self, X):
        return np.clip(self.model.predict(self._featurize(X)), SIGMA_FLOOR, SIGMA_CEIL)


MEMBER_CONFIGS = {
    "rf": build_rf_pipeline,
    "hgb_a": lambda: HGBModel(random_state=42, max_depth=8, learning_rate=0.06, max_iter=600),
    "hgb_b": lambda: HGBModel(random_state=7, max_depth=6, learning_rate=0.08, max_iter=500),
    "hgb_c": lambda: HGBModel(random_state=123, max_depth=10, learning_rate=0.05, max_iter=600),
    "hgb_d": lambda: HGBModel(random_state=2024, max_depth=5, learning_rate=0.1, max_iter=500),
}


def fit_base_models(X, y) -> dict:
    return {name: factory().fit(X, y) for name, factory in MEMBER_CONFIGS.items()}


def fit_ensemble(X_train, y_train, verbose: bool = True):
    """Fit members, NNLS blend weights and the SigmaModel.

    Weights and sigma are learned from an internal 85/15 carve-out of the *training*
    data only; the final members are then refit on all of X_train.
    """
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=RANDOM_STATE)
    val_models = fit_base_models(X_fit, y_fit)
    names = list(val_models)
    val_preds = np.column_stack([val_models[n].predict(X_val) for n in names])
    val_maes = {n: float(mean_absolute_error(y_val, val_preds[:, i])) for i, n in enumerate(names)}

    raw, _ = nnls(val_preds, y_val.to_numpy())
    total = raw.sum()
    if total <= 1e-9:  # degenerate: fall back to the single best member
        best = min(val_maes, key=val_maes.get)
        raw = np.array([1.0 if n == best else 0.0 for n in names])
        total = 1.0
    weights = {n: float(w) / float(total) for n, w in zip(names, raw)}

    val_abs_resid = np.abs(y_val.to_numpy() - val_preds @ np.array([weights[n] for n in names]))
    sigma_model = SigmaModel().fit(X_val, val_abs_resid)
    if verbose:
        print(f"    validation MAE per member: { {k: round(v, 3) for k, v in val_maes.items()} }")
        print(f"    NNLS weights: { {k: round(v, 3) for k, v in weights.items()} }")

    return fit_base_models(X_train, y_train), weights, sigma_model


def ensemble_predict(models: dict, weights: dict, X) -> np.ndarray:
    pred = np.zeros(len(X))
    for name, m in models.items():
        pred += weights.get(name, 0.0) * m.predict(X)
    return pred


def normal_cdf(x, mu, sigma):
    return norm.cdf((x - mu) / np.maximum(sigma, 1e-6))


# ------------------------------------------------------------------ calibration
def synth_students(actual, preds, sigmas, offsets=OFFSETS) -> pd.DataFrame:
    """Synthetic students at fixed offsets around each row's REAL cutoff.

    ground_truth = student_percentile >= actual cutoff.
    z uses only the model's prediction and sigma: the actual cutoff never enters it.
    """
    actual = np.asarray(actual, dtype=float)
    off = np.asarray(offsets, dtype=float)
    stu = np.clip(actual[:, None] + off[None, :], 0, 100)
    return pd.DataFrame({
        "offset": np.broadcast_to(off, stu.shape).astype(int).ravel(),
        "ground_truth": (stu >= actual[:, None]).astype(int).ravel(),
        "z": ((stu - np.asarray(preds)[:, None]) / np.asarray(sigmas)[:, None]).ravel(),
    })


def fit_calibrator(y_cal, pred_cal, sigma_cal) -> IsotonicRegression:
    cal = synth_students(y_cal, pred_cal, sigma_cal)
    return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(cal["z"], cal["ground_truth"])


# ----------------------------------------------------------------------- bundle
@dataclass
class ModelBundle:
    """Everything needed to serve predictions."""

    models: dict
    weights: dict
    sigma_model: SigmaModel
    calibrator: IsotonicRegression
    college_menu: pd.DataFrame = field(repr=False)

    def admission_probability(self, percentile: float, pool: pd.DataFrame) -> np.ndarray:
        X = pool[FEATURES]
        cutoffs = ensemble_predict(self.models, self.weights, X)
        sigmas = self.sigma_model.predict(X)
        return self.calibrator.predict((percentile - cutoffs) / sigmas)

    def recommend(self, percentile: float, category: str | None = None, branch: str | None = None,
                  gender: str | None = None, region: str | None = None,
                  min_probability: float = 0.0, top_n: int = 25) -> pd.DataFrame:
        """Rank colleges by calibrated probability that `percentile` clears the predicted cutoff.

        Results are sorted by probability, so the safest options come first.
        """
        pool = self.college_menu
        if gender:
            prefixes = eligible_category_prefixes(gender)

            def ok(cat: str) -> bool:
                if "_OTHER_" in prefixes and not cat.upper().startswith("L"):
                    return True
                return any(cat.upper().startswith(p) for p in prefixes if p != "_OTHER_")

            pool = pool[pool["category"].apply(ok)]
        elif category:
            pool = pool[pool["category"] == category]
        if branch:
            pool = pool[pool["branch"] == branch]
        if region:
            pool = pool[pool["region"] == region]
        if pool.empty:
            return pool.assign(admission_probability=pd.Series(dtype=float))
        out = pool.assign(admission_probability=self.admission_probability(percentile, pool))
        out = out[out["admission_probability"] >= min_probability]
        return out.sort_values("admission_probability", ascending=False).head(top_n)


def build_college_menu(df: pd.DataFrame) -> pd.DataFrame:
    menu = (
        df.sort_values("round")
        .groupby(["college_code", "college_name", "branch", "category", "seat_source"], as_index=False)
        .agg(final_percentile=(TARGET, "last"), last_round=("round", "last"))
    )
    menu["region"] = menu["college_name"].apply(extract_region)
    menu["gender_scope"] = menu["category"].apply(gender_scope)
    menu["allocation_level"] = menu["category"].apply(allocation_level)
    menu["round"] = menu["last_round"]
    return menu
