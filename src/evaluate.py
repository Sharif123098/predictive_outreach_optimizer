import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src import config


def ranking_metrics(y_true, score, flag_rate=None, seed=None):
    flag_rate = config.FLAG_RATE if flag_rate is None else flag_rate
    seed = config.SEED if seed is None else seed
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    n = len(y)
    k = max(1, int(round(flag_rate * n)))
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(n), -s))
    flagged = np.zeros(n, dtype=bool)
    flagged[order[:k]] = True
    positives = int(y.sum())
    prevalence = float(y.mean())
    true_positives = int(y[flagged].sum())
    precision = true_positives / k
    both_classes = positives > 0 and positives < n
    return {
        "n": int(n),
        "prevalence": prevalence,
        "flag_rate": k / n,
        "precision_at_flag": precision,
        "recall_at_flag": true_positives / positives if positives else float("nan"),
        "lift_at_flag": precision / prevalence if prevalence > 0 else float("nan"),
        "pr_auc": float(average_precision_score(y, s)) if positives else float("nan"),
        "roc_auc": float(roc_auc_score(y, s)) if both_classes else float("nan"),
    }


def threshold_metrics(y_true, prob, threshold):
    y = np.asarray(y_true).astype(int)
    flag = np.asarray(prob) >= threshold
    tp = int(((y == 1) & flag).sum())
    fp = int(((y == 0) & flag).sum())
    fn = int(((y == 1) & ~flag).sum())
    tn = int(((y == 0) & ~flag).sum())
    return {
        "threshold": float(threshold),
        "flag_rate_at_threshold": float(flag.mean()),
        "precision_at_threshold": tp / (tp + fp) if (tp + fp) else float("nan"),
        "recall_at_threshold": tp / (tp + fn) if (tp + fn) else float("nan"),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def add_slice_columns(df):
    out = df.copy()
    out["age_band"] = pd.cut(
        out["age"], bins=[-1, 17, 34, 49, 64, 200], labels=["0-17", "18-34", "35-49", "50-64", "65+"]
    ).astype(str)
    out["lead_band"] = pd.cut(
        out["lead_days"], bins=[0, 1, 7, 14, 10**6], labels=["1 day", "2-7 days", "8-14 days", "15+ days"]
    ).astype(str)
    out["handcap_any"] = np.where(out["handcap"] > 0, "yes", "no")
    return out


def slice_table(df, prob, threshold, by):
    frame = df.assign(prob=np.asarray(prob, dtype=float))
    frame["flag"] = frame["prob"] >= threshold
    rows = []
    for value, group in frame.groupby(by):
        y = group["no_show"].to_numpy()
        flag = group["flag"].to_numpy()
        positives = int(y.sum())
        rows.append({
            "slice": by,
            "value": str(value),
            "n": int(len(group)),
            "no_show_rate": float(y.mean()),
            "flag_rate": float(flag.mean()),
            "precision": float(y[flag].mean()) if flag.any() else float("nan"),
            "recall": float(y[flag].sum() / positives) if positives else float("nan"),
            "auc": float(roc_auc_score(y, group["prob"])) if 0 < positives < len(group) and len(group) >= 30 else float("nan"),
        })
    return pd.DataFrame(rows)


def all_slices(df, prob, threshold, top_neighbourhoods=10):
    frame = add_slice_columns(df).reset_index(drop=True)
    prob = np.asarray(prob, dtype=float)
    tables = []
    for by in config.SLICE_COLUMNS:
        if by == "neighbourhood":
            top = frame["neighbourhood"].value_counts().head(top_neighbourhoods).index
            mask = frame["neighbourhood"].isin(top).to_numpy()
            tables.append(slice_table(frame.loc[mask], prob[mask], threshold, by))
        else:
            tables.append(slice_table(frame, prob, threshold, by))
    return pd.concat(tables, ignore_index=True)


def gap_summary(slices, min_n=100):
    rows = []
    for by, group in slices.groupby("slice"):
        group = group[group["n"] >= min_n]
        if len(group) < 2:
            continue
        rows.append({
            "slice": by,
            "recall_gap": float(group["recall"].max() - group["recall"].min()),
            "flag_rate_gap": float(group["flag_rate"].max() - group["flag_rate"].min()),
            "precision_gap": float(group["precision"].max() - group["precision"].min()),
        })
    return pd.DataFrame(rows)


def psi(expected, actual, bins=10, eps=1e-4):
    e = pd.Series(expected).dropna()
    a = pd.Series(actual).dropna()
    if e.nunique() <= bins:
        cats = sorted(set(e.unique()) | set(a.unique()))
        pe = np.array([(e == c).mean() for c in cats])
        pa = np.array([(a == c).mean() for c in cats])
    else:
        edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        pe = np.histogram(e, bins=edges)[0] / len(e)
        pa = np.histogram(a, bins=edges)[0] / len(a)
    pe = np.clip(pe, eps, None)
    pa = np.clip(pa, eps, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi_table(train_df, test_df, columns=None):
    columns = columns or config.NUMERIC_FEATURES
    rows = [{"feature": c, "psi": psi(train_df[c], test_df[c])} for c in columns]
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def plot_confusion(y_true, flag, path):
    y = np.asarray(y_true).astype(int)
    f = np.asarray(flag).astype(int)
    matrix = np.array([[((y == 0) & (f == 0)).sum(), ((y == 0) & (f == 1)).sum()],
                       [((y == 1) & (f == 0)).sum(), ((y == 1) & (f == 1)).sum()]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["not flagged", "flagged"])
    ax.set_yticks([0, 1], ["attended", "no-show"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center")
    ax.set_title("Confusion matrix at operating point")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return matrix


def plot_importance(table, path):
    ordered = table.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(ordered["feature"], ordered["importance_mean"], xerr=ordered["importance_std"])
    ax.set_xlabel("Drop in PR-AUC when the column is shuffled")
    ax.set_title("Permutation importance (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    from src import data

    table, _ = data.build_model_table()
    predictions = pd.read_csv(config.REPORTS_DIR / "test_predictions.csv")
    with open(config.META_PATH, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    test = table[table["split"] == "test"].merge(
        predictions[["appointment_id", "probability"]], on="appointment_id", how="inner"
    )
    slices = all_slices(test, test["probability"], meta["threshold"])
    gaps = gap_summary(slices)
    train = table[table["split"] == "train"]
    drift = psi_table(train, test)
    slices.to_csv(config.REPORTS_DIR / "slice_analysis.csv", index=False)
    gaps.to_csv(config.REPORTS_DIR / "fairness_gaps.csv", index=False)
    drift.to_csv(config.REPORTS_DIR / "psi_train_vs_test.csv", index=False)
    print(gaps.to_string(index=False))
    print(drift.to_string(index=False))


if __name__ == "__main__":
    main()
