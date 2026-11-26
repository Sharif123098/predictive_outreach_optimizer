import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from src import config


def load_artifact(model_path=None, meta_path=None):
    model = joblib.load(model_path or config.MODEL_PATH)
    with open(meta_path or config.META_PATH, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    return model, meta


def score_frame(model, meta, df):
    prob = model.predict_proba(df[config.RAW_INPUT_COLUMNS])[:, 1]
    return prob, prob >= meta["threshold"]


def action_text(flag):
    if flag:
        return "Call this patient the day before: high risk of missing the appointment."
    return "No call needed: lower risk of missing the appointment."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()
    model, meta = load_artifact()
    df = pd.read_csv(args.csv_path)
    prob, flag = score_frame(model, meta, df)
    out = df.assign(probability=prob, flag=flag.astype(int))
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
