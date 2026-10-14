"""
(1) Noise floor: how much does a group's cutoff move between consecutive rounds?
(2) Calibrated probabilities (isotonic on z=(student-pred)/sigma, fit on a held-out
    calibration slice, never on test rows) and selective prediction: accuracy as a
    function of coverage when the model abstains on low-confidence cases.
All evaluation is LEAK-FREE (the predictor never sees the test row's own label).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import allocation_level, extract_region, gender_scope

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
OFFSETS = np.arange(-25, 26, 5)

df = pd.read_csv(ROOT / "data" / "combined_cutoffs.csv").dropna(subset=["percentile"]).reset_index(drop=True)
df["college_code"] = df["college_code"].astype(str)
df["region"] = df["college_name"].apply(extract_region)
df["gender_scope"] = df["category"].apply(gender_scope)
df["allocation_level"] = df["category"].apply(allocation_level)
GROUP = ["college_code", "branch", "category", "seat_source"]
df = df.sort_values(GROUP + ["round"]).reset_index(drop=True)
g = df.groupby(GROUP)
df["prev_pct"] = g["percentile"].shift(1)
df["prev_round"] = g["round"].shift(1)
df["n_prev"] = g.cumcount()
df["prev_mean"] = g["percentile"].transform(lambda s: s.shift(1).expanding().mean())
df["round_gap"] = df["round"] - df["prev_round"]

# ---------- (1) noise floor
d = df.dropna(subset=["prev_pct"])
delta = (d["percentile"] - d["prev_pct"]).abs()
print("=== NOISE FLOOR (same college/branch/category, consecutive rounds) ===")
print(f"pairs: {len(d):,}   mean |delta|: {delta.mean():.2f}   median |delta|: {delta.median():.2f}   "
      f"p75: {delta.quantile(.75):.2f}   p90: {delta.quantile(.9):.2f}")
print(f"MAE of the naive 'same as previous round' predictor: {delta.mean():.2f}")
print(f"share of pairs that move by more than 5 pts: {(delta > 5).mean():.1%},  more than 10 pts: {(delta > 10).mean():.1%}")

# ---------- model
CAT = ["branch", "category", "gender_scope", "allocation_level", "seat_source", "region"]
NUM = ["round", "college_enc", "prev_pct", "n_prev", "prev_mean", "round_gap"]


def featurize(train, others):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(train[CAT])
    gm = train["percentile"].mean()
    st = train.groupby("college_code")["percentile"].agg(["mean", "count"])
    cmap = ((st["count"] * st["mean"] + 25 * gm) / (st["count"] + 25)).to_dict()

    def b(x):
        x = x.copy()
        x["college_enc"] = x["college_code"].map(cmap).fillna(gm)
        return np.hstack([enc.transform(x[CAT]), x[NUM].to_numpy(dtype=float)])

    return b(train), [b(o) for o in others]


def hgb(**kw):
    return HistGradientBoostingRegressor(
        categorical_features=list(range(len(CAT))), early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=25, random_state=SEED, **kw)


train_all, test = train_test_split(df, test_size=0.2, random_state=SEED)
fit, calib = train_test_split(train_all, test_size=0.2, random_state=SEED)  # calibration slice, disjoint from test

Xf, (Xc, Xt) = featurize(fit, [calib, test])
point = hgb(max_iter=600, max_depth=8, learning_rate=0.06, l2_regularization=0.3).fit(Xf, fit["percentile"])
# uncertainty model trained on the calibration slice's own residuals (out-of-sample for `point`)
res_c = np.abs(calib["percentile"].to_numpy() - point.predict(Xc))
sig = hgb(max_iter=250, max_depth=5, learning_rate=0.07).fit(Xc, res_c)

pred_t = point.predict(Xt)
y_t = test["percentile"].to_numpy()
print(f"\npoint model on held-out test: MAE={mean_absolute_error(y_t, pred_t):.3f}")
sigma_t = np.clip(sig.predict(Xt), 4.0, 30.0)


def synth(y, pred, sigma):
    rows = []
    for a, p, s in zip(y, pred, sigma):
        for off in OFFSETS:
            stu = float(np.clip(a + off, 0, 100))
            rows.append((off, int(stu >= a), (stu - p) / s))
    return pd.DataFrame(rows, columns=["offset", "gt", "z"])


# calibration set: synthetic students around the CALIBRATION rows' cutoffs
pred_c = point.predict(Xc)
sigma_c = np.clip(sig.predict(Xc), 4.0, 30.0)
cal = synth(calib["percentile"].to_numpy(), pred_c, sigma_c)
iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(cal["z"], cal["gt"])

ev = synth(y_t, pred_t, sigma_t)
ev["p_gauss"] = norm.cdf(ev["z"])
ev["p_cal"] = iso.predict(ev["z"])
for col in ("p_gauss", "p_cal"):
    ev[f"hat_{col}"] = (ev[col] >= 0.5).astype(int)

print("\n=== LEAK-FREE CLASSIFICATION on held-out test (11 offsets, -25..+25) ===")
for col in ("p_gauss", "p_cal"):
    acc = (ev[f"hat_{col}"] == ev["gt"]).mean()
    nz = ev["offset"] != 0
    acc_nz = (ev.loc[nz, f"hat_{col}"] == ev.loc[nz, "gt"]).mean()
    print(f"{col:8s} acc_all={acc:.4f}  acc_excl_exact_tie={acc_nz:.4f}  AUC={roc_auc_score(ev['gt'], ev[col]):.4f}  "
          f"Brier={brier_score_loss(ev['gt'], ev[col]):.4f}")

# ---------- selective prediction (abstain when unsure), calibrated probs
print("\n=== SELECTIVE PREDICTION (calibrated probs). Model only answers when confident; coverage reported ===")
print("threshold  coverage   accuracy_on_answered")
for t in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
    conf = np.maximum(ev["p_cal"], 1 - ev["p_cal"])
    m = conf >= t
    if m.sum() == 0:
        continue
    acc = (ev.loc[m, "hat_p_cal"] == ev.loc[m, "gt"]).mean()
    print(f"  p>={t:.2f}    {m.mean():7.1%}    {acc:.4f}")
print("\nreference: share of cases at each offset that are answered at p>=0.90:")
conf = np.maximum(ev["p_cal"], 1 - ev["p_cal"])
print((conf >= 0.9).groupby(ev["offset"]).mean().round(3).to_dict())
