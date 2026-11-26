import hashlib
import json

import numpy as np
import pandas as pd

from src import config
from src.features import decision_time, make_features, to_naive

RENAME = {
    "PatientId": "patient_id",
    "AppointmentID": "appointment_id",
    "Gender": "gender",
    "ScheduledDay": "scheduled_day",
    "AppointmentDay": "appointment_day",
    "Age": "age",
    "Neighbourhood": "neighbourhood",
    "Scholarship": "scholarship",
    "Hipertension": "hipertension",
    "Diabetes": "diabetes",
    "Alcoholism": "alcoholism",
    "Handcap": "handcap",
    "SMS_received": "sms_received",
    "No-show": "no_show",
}

BINARY_COLUMNS = ["scholarship", "hipertension", "diabetes", "alcoholism", "sms_received"]


def file_md5(path=None):
    path = path or config.DATA_RAW
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(path=None):
    path = path or config.DATA_RAW
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    missing = sorted(set(RENAME) - set(df.columns))
    if missing:
        raise ValueError(f"Columns missing from the raw file: {missing}")
    return df.rename(columns=RENAME)[list(RENAME.values())]


def clean(raw):
    report = {"rows_raw": int(len(raw))}
    df = raw.copy()
    df["patient_id"] = pd.to_numeric(df["patient_id"], errors="raise").round().astype("int64")
    df["appointment_id"] = pd.to_numeric(df["appointment_id"], errors="raise").astype("int64")
    df["scheduled_day"] = to_naive(df["scheduled_day"])
    df["appointment_day"] = to_naive(df["appointment_day"]).dt.normalize()
    df["gender"] = df["gender"].astype(str).str.strip()
    df["neighbourhood"] = df["neighbourhood"].astype(str).str.strip()
    df["age"] = pd.to_numeric(df["age"], errors="raise")
    df["handcap"] = pd.to_numeric(df["handcap"], errors="raise").astype(int)
    for col in BINARY_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)
    target = df["no_show"].astype(str).str.strip().map({"Yes": 1, "No": 0})
    if target.isna().any():
        raise ValueError("no_show contains values other than Yes and No")
    df["no_show"] = target.astype(int)

    duplicated = df.duplicated("appointment_id", keep="first")
    report["rows_duplicate_appointment_id"] = int(duplicated.sum())
    df = df.loc[~duplicated]

    bad_age = (df["age"] < config.AGE_MIN) | (df["age"] > config.AGE_MAX)
    report["rows_invalid_age"] = int(bad_age.sum())
    df = df.loc[~bad_age].copy()
    df["age"] = df["age"].astype(int)

    booked_after_appointment_date = df["scheduled_day"].dt.normalize() > df["appointment_day"]
    report["rows_booked_after_appointment_date"] = int(booked_after_appointment_date.sum())
    report["rows_clean"] = int(len(df))
    return df.reset_index(drop=True), report


def add_history(df):
    out = df.copy()
    pid = out["patient_id"].to_numpy()
    days = out["appointment_day"].to_numpy().astype("datetime64[D]").astype("int64")
    target = out["no_show"].to_numpy()
    order = np.lexsort((days, pid))
    pid_s, days_s, target_s = pid[order], days[order], target[order]
    starts = np.flatnonzero(np.r_[True, pid_s[1:] != pid_s[:-1]])
    ends = np.r_[starts[1:], len(pid_s)]
    prior_n = np.zeros(len(out), dtype="int64")
    prior_pos = np.zeros(len(out), dtype="int64")
    for start, end in zip(starts, ends):
        group_days = days_s[start:end]
        cumulative = np.r_[0, np.cumsum(target_s[start:end])]
        count = np.searchsorted(group_days, group_days - 1, side="left")
        prior_n[start:end] = count
        prior_pos[start:end] = cumulative[count]
    restored_n = np.zeros(len(out), dtype="int64")
    restored_pos = np.zeros(len(out), dtype="int64")
    restored_n[order] = prior_n
    restored_pos[order] = prior_pos
    out["prior_appts"] = restored_n
    out["prior_noshows"] = restored_pos
    return out


def assign_split(df):
    counts = df.groupby("appointment_day").size().sort_index()
    share = counts.cumsum() / counts.sum()
    train_end = share.index[int(np.argmax(share.to_numpy() >= 1 - config.TEST_SHARE))]
    split = np.where(df["appointment_day"] <= train_end, "train", "test")
    if (split == "test").sum() == 0 or (split == "train").sum() == 0:
        raise ValueError("Time split produced an empty train or test set")
    return split, train_end


def build_model_table(path=None):
    raw = load_raw(path)
    cleaned, report = clean(raw)
    full = add_history(cleaned)
    full["decision_time"] = decision_time(full["appointment_day"])
    eligible = full["scheduled_day"] <= full["decision_time"]
    excluded = full.loc[~eligible]
    report["rows_not_booked_by_decision_time"] = int((~eligible).sum())
    report["no_show_rate_not_booked_by_decision_time"] = (
        float(excluded["no_show"].mean()) if len(excluded) else None
    )
    table = full.loc[eligible].copy()
    extra = make_features(table).drop(columns=[c for c in config.NUMERIC_FEATURES if c in table.columns])
    table = pd.concat([table, extra], axis=1)
    split, train_end = assign_split(table)
    table["split"] = split
    table = table.sort_values(["appointment_day", "scheduled_day", "appointment_id"]).reset_index(drop=True)
    report["rows_model_table"] = int(len(table))
    report["rows_train"] = int((table["split"] == "train").sum())
    report["rows_test"] = int((table["split"] == "test").sum())
    report["train_end_date"] = str(pd.Timestamp(train_end).date())
    report["first_appointment_date"] = str(table["appointment_day"].min().date())
    report["last_appointment_date"] = str(table["appointment_day"].max().date())
    return table, report


def forward_day_splits(days, n_splits):
    days = pd.Series(pd.to_datetime(days)).reset_index(drop=True)
    unique_days = np.sort(days.unique())
    chunks = np.array_split(unique_days, n_splits + 1)
    splits = []
    for k in range(n_splits):
        train_days = np.concatenate(chunks[: k + 1])
        val_days = chunks[k + 1]
        train_idx = np.flatnonzero(days.isin(train_days).to_numpy())
        val_idx = np.flatnonzero(days.isin(val_days).to_numpy())
        splits.append((train_idx, val_idx))
    return splits


def save_processed(table, report):
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(config.DATA_PROCESSED_DIR / "appointments_model_table.csv", index=False)
    with open(config.DATA_PROCESSED_DIR / "cleaning_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
