import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src import config


def to_naive(series):
    return pd.to_datetime(series, utc=True).dt.tz_localize(None)


def decision_time(appointment_day):
    day = pd.to_datetime(appointment_day).dt.normalize()
    return day - pd.Timedelta(days=1) + pd.Timedelta(hours=config.DECISION_HOUR)


def make_features(df):
    sched = to_naive(df["scheduled_day"])
    appt = to_naive(df["appointment_day"]).dt.normalize()
    decision = decision_time(appt)
    out = pd.DataFrame(index=df.index)
    for col in ["age", "scholarship", "hipertension", "diabetes", "alcoholism", "prior_appts", "prior_noshows"]:
        out[col] = pd.to_numeric(df[col])
    out["lead_days"] = (appt - sched.dt.normalize()).dt.days
    out["lead_hours"] = (decision - sched).dt.total_seconds() / 3600.0
    out["scheduled_hour"] = sched.dt.hour
    out["scheduled_weekday"] = sched.dt.dayofweek
    out["appointment_weekday"] = appt.dt.dayofweek
    out["has_history"] = (out["prior_appts"] > 0).astype(int)
    out["prior_noshow_rate"] = out["prior_noshows"] / out["prior_appts"].replace(0, np.nan)
    return out[config.NUMERIC_FEATURES]


def build_pipeline(estimator, columns=None, linear=False):
    cols = list(columns or config.NUMERIC_FEATURES)
    if linear:
        numeric = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
    else:
        numeric = "passthrough"
    prep = ColumnTransformer([("num", numeric, cols)], remainder="drop")
    return Pipeline([
        ("features", FunctionTransformer(make_features, validate=False)),
        ("prep", prep),
        ("model", estimator),
    ])
