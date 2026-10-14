import numpy as np
import pandas as pd

from src.model import (
    FEATURES, TARGET, CollegeTargetEncoder, HGBModel, SIGMA_CEIL, SIGMA_FLOOR, SigmaModel,
    fit_calibrator, load_data, synth_students,
)


def test_synth_students_ground_truth_and_no_label_in_z():
    actual = np.array([60.0, 80.0])
    preds = np.array([50.0, 90.0])
    sig = np.array([10.0, 10.0])
    ev = synth_students(actual, preds, sig, offsets=[-5, 0, 5])
    assert list(ev["ground_truth"]) == [0, 1, 1, 0, 1, 1]
    # z depends on student, prediction and sigma only: row 0, offset -5 -> student 55 -> (55-50)/10
    assert ev["z"].iloc[0] == 0.5


def test_target_encoder_unseen_falls_back_to_global_mean():
    codes = pd.Series(["a", "a", "b"])
    enc = CollegeTargetEncoder(smoothing=1.0).fit(codes, [10.0, 20.0, 60.0])
    out = enc.transform(pd.Series(["a", "zzz"]))
    assert out[1] == enc.global_mean_
    assert 10.0 < out[0] < 60.0


def test_calibrator_is_monotone_probability():
    rng = np.random.default_rng(0)
    y = rng.uniform(40, 99, 300)
    pred = y + rng.normal(0, 8, 300)
    iso = fit_calibrator(y, pred, np.full(300, 8.0))
    p = iso.predict(np.linspace(-4, 4, 50))
    assert np.all(np.diff(p) >= -1e-12) and p.min() >= 0 and p.max() <= 1


def test_small_fit_on_real_data_beats_mean_baseline():
    df = load_data().sample(6000, random_state=0)
    assert "sml_rank" not in FEATURES  # leakage guard
    train, test = df.iloc[:4500], df.iloc[4500:]
    m = HGBModel(max_iter=150).fit(train[FEATURES], train[TARGET])
    mae = np.abs(m.predict(test[FEATURES]) - test[TARGET]).mean()
    baseline = np.abs(train[TARGET].mean() - test[TARGET]).mean()
    assert mae < baseline
    s = SigmaModel().fit(train[FEATURES], np.abs(train[TARGET] - m.predict(train[FEATURES])))
    sig = s.predict(test[FEATURES])
    assert sig.min() >= SIGMA_FLOOR and sig.max() <= SIGMA_CEIL
