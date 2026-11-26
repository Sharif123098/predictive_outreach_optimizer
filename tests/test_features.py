import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src import config, data
from src.evaluate import psi, ranking_metrics
from src.features import build_pipeline, make_features


def make_raw(rows):
    return pd.DataFrame(rows)


def test_lead_features_are_relative_to_decision_time():
    df = pd.DataFrame([{
        "age": 30, "scholarship": 0, "hipertension": 0, "diabetes": 0, "alcoholism": 0,
        "scheduled_day": "2016-05-02T10:00:00Z", "appointment_day": "2016-05-04T00:00:00Z",
        "prior_appts": 0, "prior_noshows": 0,
    }])
    feats = make_features(df)
    assert feats.loc[0, "lead_days"] == 2
    assert feats.loc[0, "lead_hours"] == 31.0
    assert feats.loc[0, "appointment_weekday"] == 2
    assert np.isnan(feats.loc[0, "prior_noshow_rate"])
    assert feats.loc[0, "has_history"] == 0


def test_history_excludes_same_and_previous_day_outcomes():
    base = pd.Timestamp("2016-05-02")
    df = pd.DataFrame({
        "patient_id": [1, 1, 1, 1],
        "appointment_id": [1, 2, 3, 4],
        "appointment_day": [base, base + pd.Timedelta(days=2), base + pd.Timedelta(days=3), base + pd.Timedelta(days=4)],
        "no_show": [1, 1, 0, 0],
    })
    out = data.add_history(df).sort_values("appointment_id")
    assert out["prior_appts"].tolist() == [0, 1, 1, 2]
    assert out["prior_noshows"].tolist() == [0, 1, 1, 2]


def test_history_is_separate_per_patient():
    base = pd.Timestamp("2016-05-02")
    df = pd.DataFrame({
        "patient_id": [1, 2, 1, 2],
        "appointment_id": [1, 2, 3, 4],
        "appointment_day": [base, base, base + pd.Timedelta(days=5), base + pd.Timedelta(days=5)],
        "no_show": [1, 0, 0, 0],
    })
    out = data.add_history(df).sort_values("appointment_id")
    assert out["prior_noshows"].tolist() == [0, 0, 1, 0]


def test_forward_day_splits_never_train_on_the_future():
    days = pd.Series(pd.date_range("2016-05-01", periods=30).repeat(3))
    for train_idx, val_idx in data.forward_day_splits(days, 5):
        assert days.iloc[train_idx].max() < days.iloc[val_idx].min()


def test_pipeline_fits_and_predicts_on_raw_columns():
    rng = np.random.default_rng(0)
    n = 300
    appt = pd.Timestamp("2016-05-10") + pd.to_timedelta(rng.integers(0, 10, n), unit="D")
    df = pd.DataFrame({
        "age": rng.integers(0, 90, n),
        "scholarship": rng.integers(0, 2, n),
        "hipertension": rng.integers(0, 2, n),
        "diabetes": rng.integers(0, 2, n),
        "alcoholism": rng.integers(0, 2, n),
        "scheduled_day": appt - pd.to_timedelta(rng.integers(1, 20, n), unit="D"),
        "appointment_day": appt,
        "prior_appts": rng.integers(0, 4, n),
        "prior_noshows": 0,
    })
    y = pd.Series(rng.integers(0, 2, n))
    pipe = build_pipeline(LogisticRegression(max_iter=500), linear=True).fit(df[config.RAW_INPUT_COLUMNS], y)
    prob = pipe.predict_proba(df[config.RAW_INPUT_COLUMNS])[:, 1]
    assert prob.shape == (n,)
    assert ((prob >= 0) & (prob <= 1)).all()


def test_ranking_metrics_on_perfect_scores():
    y = np.array([1] * 20 + [0] * 80)
    metrics = ranking_metrics(y, y.astype(float), flag_rate=0.2)
    assert metrics["precision_at_flag"] == 1.0
    assert metrics["recall_at_flag"] == 1.0
    assert abs(metrics["lift_at_flag"] - 5.0) < 1e-9


def test_psi_is_zero_for_identical_samples():
    values = np.random.default_rng(0).normal(size=1000)
    assert psi(values, values) < 1e-9
