"""
Exploration: (1) how much of the earlier 90% was due to an evaluation leak, and
(2) how much a legitimately-available lag feature (the same college/branch/
category's cutoff in an EARLIER round) can help.

Leak being tested: the earlier classification eval used
    blended = (actual_cutoff + model_pred) / 2
where actual_cutoff is the held-out label itself. That feeds the answer into the
predictor and halves its effective error. A leak-free predictor may only use
model_pred (which uses features available before the round is published).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import allocation_level, extract_region, gender_scope

ROOT = Path(__file__).resolve().parent.parent
OFFSETS = np.arange(-25, 26, 5)
SEED = 42

df = pd.read_csv(ROOT / "data" / "combined_cutoffs.csv").dropna(subset=["percentile"]).reset_index(drop=True)
df["college_code"] = df["college_code"].astype(str)
df["region"] = df["college_name"].apply(extract_region)
df["gender_scope"] = df["category"].apply(gender_scope)
df["allocation_level"] = df["category"].apply(allocation_level)

# ---- lag features: only information from STRICTLY EARLIER rounds of the same group
GROUP = ["college_code", "branch", "category", "seat_source"]
df = df.sort_values(GROUP + ["round"]).reset_index(drop=True)
g = df.groupby(GROUP)
df["prev_pct"] = g["percentile"].shift(1)
df["prev_round"] = g["round"].shift(1)
df["n_prev"] = g.cumcount()
df["prev_mean"] = g["percentile"].transform(lambda s: s.shift(1).expanding().mean())
df["round_gap"] = df["round"] - df["prev_round"]
print(f"rows with a lag available: {df['prev_pct'].notna().mean():.1%}")

CAT = ["branch", "category", "gender_scope", "allocation_level", "seat_source", "region"]
NUM_BASE = ["round", "college_enc"]
NUM_LAG = ["prev_pct", "n_prev", "prev_mean", "round_gap"]


def make_xy(train, test, use_lag):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(train[CAT])
    smooth = 25.0
    gm = train["percentile"].mean()
    st = train.groupby("college_code")["percentile"].agg(["mean", "count"])
    cmap = ((st["count"] * st["mean"] + smooth * gm) / (st["count"] + smooth)).to_dict()

    def build(d):
        d = d.copy()
        d["college_enc"] = d["college_code"].map(cmap).fillna(gm)
        cols = NUM_BASE + (NUM_LAG if use_lag else [])
        return np.hstack([enc.transform(d[CAT]), d[cols].to_numpy(dtype=float)])

    return build(train), build(test)


def fit_predict(train, test, use_lag):
    Xtr, Xte = make_xy(train, test, use_lag)
    m = HistGradientBoostingRegressor(
        max_iter=600, max_depth=8, learning_rate=0.06, l2_regularization=0.3,
        categorical_features=list(range(len(CAT))), early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=25, random_state=SEED,
    ).fit(Xtr, train["percentile"])
    # per-row uncertainty model on |residual| from out-of-fold-ish train residuals
    tr_pred = m.predict(Xtr)
    sm = HistGradientBoostingRegressor(
        max_iter=250, max_depth=5, learning_rate=0.07, early_stopping=True, random_state=SEED,
        categorical_features=list(range(len(CAT))),
    ).fit(Xtr, np.abs(train["percentile"].to_numpy() - tr_pred))
    return m.predict(Xte), np.clip(sm.predict(Xte), 4.0, 30.0)


def classify(y_true, pred, sigma, leaky):
    rows = []
    for a, p, s in zip(y_true, pred, sigma):
        mu = (a + p) / 2.0 if leaky else p
        for off in OFFSETS:
            stu = float(np.clip(a + off, 0, 100))
            rows.append((off, int(stu >= a), norm.cdf((stu - mu) / s)))
    e = pd.DataFrame(rows, columns=["offset", "gt", "prob"])
    e["pred"] = (e["prob"] >= 0.5).astype(int)
    e["ok"] = (e["pred"] == e["gt"]).astype(int)
    return e


def report(tag, e):
    nz = e[e["offset"] != 0]
    far = e[e["offset"].abs() >= 10]
    print(f"{tag:42s} acc_all={e['ok'].mean():.4f}  acc_excl_tie(offset!=0)={nz['ok'].mean():.4f}  "
          f"acc_|off|>=10={far['ok'].mean():.4f}  AUC={roc_auc_score(e['gt'], e['prob']):.4f}")
    print("    by offset:", {int(k): round(v, 3) for k, v in e.groupby('offset')['ok'].mean().items()})


# ---------------- random split
tr, te = train_test_split(df, test_size=0.2, random_state=SEED)
for use_lag in (False, True):
    pred, sigma = fit_predict(tr, te, use_lag)
    y = te["percentile"].to_numpy()
    print(f"\n=== RANDOM SPLIT  use_lag={use_lag}  MAE={mean_absolute_error(y, pred):.3f}  R2={r2_score(y, pred):.4f}")
    report("LEAKY eval (blend includes the label)", classify(y, pred, sigma, leaky=True))
    report("LEAK-FREE eval (pred only)", classify(y, pred, sigma, leaky=False))

# ---------------- temporal split: train on rounds 1-3, predict round 4 (the honest 'future round' test)
tr_t, te_t = df[df["round"] <= 3], df[df["round"] == 4]
for use_lag in (False, True):
    pred, sigma = fit_predict(tr_t, te_t, use_lag)
    y = te_t["percentile"].to_numpy()
    print(f"\n=== TEMPORAL (train R1-3 -> test R4)  use_lag={use_lag}  n={len(te_t)}  "
          f"MAE={mean_absolute_error(y, pred):.3f}  R2={r2_score(y, pred):.4f}")
    report("LEAK-FREE eval (pred only)", classify(y, pred, sigma, leaky=False))
    has = te_t["prev_pct"].notna().to_numpy()
    if use_lag and has.any():
        print(f"    subset with lag available ({has.mean():.1%} of R4): MAE={mean_absolute_error(y[has], pred[has]):.3f}")
        report("    LEAK-FREE eval, lag-available subset", classify(y[has], pred[has], sigma[has], leaky=False))
