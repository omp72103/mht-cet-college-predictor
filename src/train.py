"""
Train and evaluate the MHT-CET college predictor.

    cd ml
    python -m src.train                     # quick: one fit + leak-free evaluation
    python -m src.train --robustness        # adds 5-fold CV, unseen-college GroupKFold, temporal split

Outputs (all produced by this run, nothing is hard-coded):
    models/bundle.joblib      trained model bundle (git-ignored; regenerate with this script)
    metrics.json              every reported number, written by this script
    docs/figures/*.png        figures drawn from this run's predictions

Evaluation protocol is LEAK-FREE: for each held-out row, synthetic students are placed at
percentile offsets of -25..+25 around that row's real cutoff and the model must say whether
they clear it, using only its own predicted cutoff and sigma. The held-out row's cutoff is
never an input to the prediction.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import GroupKFold, KFold, train_test_split

from src.model import (
    FEATURES, RANDOM_STATE, TARGET, ModelBundle, build_college_menu, ensemble_predict,
    fit_calibrator, fit_ensemble, load_data, normal_cdf, synth_students,
)

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "docs" / "figures"
MODEL_DIR = ROOT / "models"
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)


def fit_with_calibration(X, y, verbose=True):
    """Fit on 80% of (X, y); keep the other 20% as a disjoint calibration slice."""
    X_fit, X_cal, y_fit, y_cal = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    models, weights, sigma_model = fit_ensemble(X_fit, y_fit, verbose=verbose)
    iso = fit_calibrator(y_cal, ensemble_predict(models, weights, X_cal), sigma_model.predict(X_cal))
    return models, weights, sigma_model, iso


def classify(y_true, pred, sigma, iso) -> pd.DataFrame:
    ev = synth_students(y_true, pred, sigma)
    ev["prob_gauss"] = normal_cdf(ev["z"], 0.0, 1.0)
    ev["prob"] = iso.predict(ev["z"])
    ev["predicted"] = (ev["prob"] >= 0.5).astype(int)
    ev["correct"] = (ev["predicted"] == ev["ground_truth"]).astype(int)
    return ev


def summarise(ev: pd.DataFrame) -> dict:
    gt, pr = ev["ground_truth"], ev["predicted"]
    nz = ev["offset"] != 0
    far = ev["offset"].abs() >= 10
    conf = np.maximum(ev["prob"], 1 - ev["prob"])
    selective = []
    for t in THRESHOLDS:
        m = conf >= t
        if m.sum():
            selective.append({"threshold": t, "coverage": float(m.mean()),
                              "accuracy": float((pr[m] == gt[m]).mean())})
    return {
        "accuracy": float(accuracy_score(gt, pr)),
        "accuracy_excluding_exact_tie": float(accuracy_score(gt[nz], pr[nz])),
        "accuracy_abs_offset_ge_10": float(accuracy_score(gt[far], pr[far])),
        "precision": float(precision_score(gt, pr, zero_division=0)),
        "recall": float(recall_score(gt, pr, zero_division=0)),
        "f1": float(f1_score(gt, pr, zero_division=0)),
        "roc_auc": float(roc_auc_score(gt, ev["prob"])),
        "brier_gaussian": float(brier_score_loss(gt, ev["prob_gauss"])),
        "brier_calibrated": float(brier_score_loss(gt, ev["prob"])),
        "accuracy_by_offset": {str(k): float(v) for k, v in ev.groupby("offset")["correct"].mean().items()},
        "selective_prediction": selective,
        "n_synthetic_samples": int(len(ev)),
    }


def regression_metrics(y_true, pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(mean_squared_error(y_true, pred) ** 0.5),
        "r2": float(r2_score(y_true, pred)),
        "n_test": int(len(y_true)),
    }


def noise_floor(df: pd.DataFrame) -> dict:
    """How much does the same college/branch/category cutoff move between consecutive rounds?"""
    d = df.sort_values(["college_code", "branch", "category", "seat_source", "round"])
    prev = d.groupby(["college_code", "branch", "category", "seat_source"])[TARGET].shift(1)
    delta = (d[TARGET] - prev).abs().dropna()
    return {"pairs": int(len(delta)), "mean_abs_change": float(delta.mean()),
            "median_abs_change": float(delta.median()), "p90_abs_change": float(delta.quantile(0.9)),
            "naive_previous_round_mae": float(delta.mean())}


def make_figures(y_test, pred_test, ev, summary):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, pred_test, s=3, alpha=0.25)
    plt.plot([0, 100], [0, 100], "r--", lw=1)
    plt.xlabel("Actual cutoff percentile"); plt.ylabel("Predicted cutoff percentile")
    plt.title("Held-out test: predicted vs actual cutoff")
    plt.tight_layout(); plt.savefig(FIG_DIR / "pred_vs_actual.png", dpi=130); plt.close()

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix(ev["ground_truth"], ev["predicted"]),
                           display_labels=["Not admitted", "Admitted"]).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion matrix (leak-free)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=130); plt.close()

    fpr, tpr, _ = roc_curve(ev["ground_truth"], ev["prob"])
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {summary['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend()
    plt.title("ROC curve (leak-free)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "roc_curve.png", dpi=130); plt.close()

    by = pd.Series(summary["accuracy_by_offset"]); by.index = by.index.astype(int)
    plt.figure(figsize=(7, 4))
    plt.plot(by.index, by.values, marker="o")
    plt.axvline(0, color="gray", ls="--", lw=1)
    plt.xlabel("Student percentile minus actual cutoff"); plt.ylabel("Accuracy")
    plt.title("Accuracy vs distance from the cutoff")
    plt.tight_layout(); plt.savefig(FIG_DIR / "accuracy_by_offset.png", dpi=130); plt.close()

    sel = pd.DataFrame(summary["selective_prediction"])
    plt.figure(figsize=(6, 4))
    plt.plot(sel["coverage"] * 100, sel["accuracy"] * 100, marker="o")
    plt.axhline(95, color="gray", ls="--", lw=1)
    plt.gca().invert_xaxis()
    plt.xlabel("Coverage: % of cases answered"); plt.ylabel("Accuracy on answered cases (%)")
    plt.title("Selective prediction: accuracy vs coverage")
    plt.tight_layout(); plt.savefig(FIG_DIR / "selective_prediction.png", dpi=130); plt.close()


def run_robustness(df, X, y) -> dict:
    out = {}
    t = time.time()
    print("\n[robustness] 5-fold CV ...")
    scores = []
    for i, (tr, te) in enumerate(KFold(5, shuffle=True, random_state=RANDOM_STATE).split(X), 1):
        m, w, _ = fit_ensemble(X.iloc[tr], y.iloc[tr], verbose=False)
        scores.append(float(mean_absolute_error(y.iloc[te], ensemble_predict(m, w, X.iloc[te]))))
        print(f"    fold {i}: MAE={scores[-1]:.3f}")
    out["cv_mae_5fold"] = {"scores": scores, "mean": float(np.mean(scores)), "std": float(np.std(scores))}

    print("[robustness] GroupKFold by college_code (unseen colleges) ...")
    yt, yp = [], []
    for i, (tr, te) in enumerate(GroupKFold(5).split(X, y, df["college_code"]), 1):
        m, w, _ = fit_ensemble(X.iloc[tr], y.iloc[tr], verbose=False)
        yt.extend(y.iloc[te]); yp.extend(ensemble_predict(m, w, X.iloc[te]))
        print(f"    fold {i} done")
    out["group_split_unseen_colleges"] = {"mae": float(mean_absolute_error(yt, yp)), "r2": float(r2_score(yt, yp))}

    print("[robustness] temporal split: train rounds 1-3, test round 4 ...")
    tr_mask, te_mask = (df["round"] <= 3).to_numpy(), (df["round"] == 4).to_numpy()
    if te_mask.any() and tr_mask.any():
        models, weights, sigma, iso = fit_with_calibration(X[tr_mask], y[tr_mask], verbose=False)
        pred = ensemble_predict(models, weights, X[te_mask])
        ev = classify(y[te_mask].to_numpy(), pred, sigma.predict(X[te_mask]), iso)
        s = summarise(ev)
        out["temporal_round4"] = {**regression_metrics(y[te_mask], pred),
                                  "accuracy": s["accuracy"], "accuracy_excluding_exact_tie": s["accuracy_excluding_exact_tie"],
                                  "roc_auc": s["roc_auc"]}
    print(f"[robustness] done in {time.time() - t:.0f}s")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robustness", action="store_true", help="also run 5-fold CV, GroupKFold and temporal split (slow)")
    ap.add_argument("--no-save-model", action="store_true", help="skip writing models/bundle.joblib")
    args = ap.parse_args()

    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    print(f"{len(df):,} rows, {df['college_code'].nunique()} colleges, rounds {sorted(df['round'].unique())}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    print("Fitting ensemble (64% fit / 16% calibration / 20% untouched test) ...")
    models, weights, sigma_model, iso = fit_with_calibration(X_tr, y_tr)

    pred = ensemble_predict(models, weights, X_te)
    sigma = sigma_model.predict(X_te)
    reg = regression_metrics(y_te, pred)
    ev = classify(y_te.to_numpy(), pred, sigma, iso)
    summary = summarise(ev)
    print(f"\nRegression: MAE={reg['mae']:.3f}  R2={reg['r2']:.4f}")
    print(f"Admission classification (leak-free): accuracy={summary['accuracy']:.4f} "
          f"(excl. exact tie {summary['accuracy_excluding_exact_tie']:.4f})  AUC={summary['roc_auc']:.4f}")
    for r in summary["selective_prediction"]:
        print(f"  confidence>={r['threshold']:.2f}: coverage={r['coverage']:.1%}  accuracy={r['accuracy']:.4f}")

    metrics = {
        "evaluation": "leak-free: predictor never sees the held-out row's own cutoff",
        "data": {"rows": int(len(df)), "colleges": int(df["college_code"].nunique())},
        "random_split": {**reg, "learned_sigma_mean": float(sigma.mean())},
        "admission_classification": summary,
        "noise_floor": noise_floor(df),
        "blend_weights": weights,
    }
    if args.robustness:
        metrics.update(run_robustness(df, X, y))

    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    make_figures(y_te, pred, ev, summary)
    print("\nWrote metrics.json and docs/figures/*.png")

    if not args.no_save_model:
        MODEL_DIR.mkdir(exist_ok=True)
        bundle = ModelBundle(models, weights, sigma_model, iso, build_college_menu(df))
        joblib.dump(bundle, MODEL_DIR / "bundle.joblib", compress=3)
        print("Wrote models/bundle.joblib")


if __name__ == "__main__":
    main()
