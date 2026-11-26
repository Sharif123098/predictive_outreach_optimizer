import json
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib

matplotlib.use("Agg")
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src import config, data, evaluate
from src.features import build_pipeline, make_features

METRIC_KEYS = ["pr_auc", "roc_auc", "precision_at_flag", "recall_at_flag", "lift_at_flag"]


def git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=config.ROOT, capture_output=True, text=True, check=False
        )
    except FileNotFoundError:
        return "git-not-installed"
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else "no-commit"


def setup_tracking():
    mlflow.set_tracking_uri(config.TRACKING_URI)
    if mlflow.get_experiment_by_name(config.EXPERIMENT_NAME) is None:
        mlflow.create_experiment(config.EXPERIMENT_NAME, artifact_location=config.ARTIFACT_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)


def cross_validate_scores(score_fn, y, splits):
    folds, oof = [], []
    for train_idx, val_idx in splits:
        scores = np.asarray(score_fn(train_idx, val_idx), dtype=float)
        folds.append(evaluate.ranking_metrics(y.iloc[val_idx], scores))
        oof.append(scores)
    return folds, np.concatenate(oof)


def model_score_fn(pipeline, X, y):
    def score(train_idx, val_idx):
        fitted = clone(pipeline).fit(X.iloc[train_idx], y.iloc[train_idx])
        return fitted.predict_proba(X.iloc[val_idx])[:, 1]

    return score


def summarize(folds):
    summary = {}
    for key in METRIC_KEYS:
        values = np.array([fold[key] for fold in folds], dtype=float)
        summary[f"cv_{key}_mean"] = float(values.mean())
        summary[f"cv_{key}_std"] = float(values.std(ddof=1))
    return summary


def log_cv_run(run_name, params, folds, tags, artifacts=()):
    summary = summarize(folds)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(tags)
        mlflow.log_params({k: str(v) for k, v in params.items()})
        for step, fold in enumerate(folds):
            for key in METRIC_KEYS:
                mlflow.log_metric(f"cv_fold_{key}", fold[key], step=step)
        mlflow.log_metrics(summary)
        for path in artifacts:
            mlflow.log_artifact(str(path))
        return run.info.run_id, summary


def tune(name, pipeline, space, kind, X, y, splits):
    if kind == "grid":
        search = GridSearchCV(pipeline, space, scoring="average_precision", cv=splits, n_jobs=1, refit=False)
    else:
        search = RandomizedSearchCV(
            pipeline, space, n_iter=config.N_ITER_HGB, scoring="average_precision",
            cv=splits, n_jobs=1, random_state=config.SEED, refit=False,
        )
    search.fit(X, y)
    results = pd.DataFrame(search.cv_results_)
    best_index = int(np.argmin(results["rank_test_score"].to_numpy()))
    best_params = results.loc[best_index, "params"]
    path = config.REPORTS_DIR / f"search_{name}.csv"
    results.drop(columns=["params"]).to_csv(path, index=False)
    return best_params, path


def library_versions():
    names = ["scikit-learn", "pandas", "numpy", "joblib", "mlflow"]
    versions = {name: metadata.version(name) for name in names}
    versions["python"] = sys.version.split()[0]
    return versions


def build_recommendation(test_start, test_end, model, rule, criteria, flagged_per_day, break_even):
    verdict = criteria["verdict"]
    lines = [
        f"Recommendation: {verdict}.",
        "",
        (
            f"On the held-out later period ({test_start} to {test_end}), flagging the top "
            f"{model['flag_rate']:.0%} of next-day appointments reaches {model['precision_at_flag']:.1%} "
            f"no-shows among the patients called, which is {model['lift_at_flag']:.2f}x the base rate of "
            f"{model['prevalence']:.1%}. The lead-time rule reaches {rule['precision_at_flag']:.1%} at the same flag rate."
        ),
        "",
        (
            f"Charter check: beats the rule = {criteria['beats_rule']}; lift at least {config.MIN_LIFT} = "
            f"{criteria['lift_ok']}; kill criterion (a) triggered = {criteria['kill_a']}; "
            f"kill criterion (b) triggered = {criteria['kill_b']}. Kill criterion (c) needs a manual check."
        ),
        "",
    ]
    if break_even is None:
        lines.append(
            f"At about {flagged_per_day:.0f} calls per day, calling cannot pay off when a missed slot is assumed "
            f"to cost {config.COST_RATIO:.0f} calls."
        )
    else:
        lines.append(
            f"At about {flagged_per_day:.0f} calls per day, calling pays off only if at least {break_even:.0%} of "
            f"reached no-shows then attend, when a missed slot is assumed to cost {config.COST_RATIO:.0f} calls."
        )
    return "\n".join(lines) + "\n"


def main():
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    session_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    commit = git_commit()
    setup_tracking()

    md5 = data.file_md5()
    if config.EXPECTED_DATA_MD5 and md5 != config.EXPECTED_DATA_MD5:
        raise SystemExit(f"Data hash mismatch: expected {config.EXPECTED_DATA_MD5}, got {md5}")

    table, report = data.build_model_table()
    report["data_md5"] = md5
    data.save_processed(table, report)

    train_df = table[table["split"] == "train"].reset_index(drop=True)
    test_df = table[table["split"] == "test"].reset_index(drop=True)
    X_train, y_train = train_df[config.RAW_INPUT_COLUMNS], train_df[config.TARGET]
    X_test, y_test = test_df[config.RAW_INPUT_COLUMNS], test_df[config.TARGET]

    splits = data.forward_day_splits(train_df["appointment_day"], config.N_SPLITS)
    for train_idx, val_idx in splits:
        if len(train_idx) == 0 or len(val_idx) == 0:
            raise SystemExit("A cross-validation fold is empty; check the number of appointment days")

    tags = {"session_id": session_id, "data_md5": md5, "git_commit": commit}
    common = {
        "seed": config.SEED,
        "split_strategy": "time-based: train on earlier appointment days, test on later days",
        "cv_strategy": f"forward-chaining by appointment day, {config.N_SPLITS} folds",
        "flag_rate": config.FLAG_RATE,
        "decision_point": f"{config.DECISION_HOUR}:00 on the day before the appointment",
        "train_rows": len(train_df),
    }

    features_all = make_features(X_train)
    results = {}

    def constant_scores(train_idx, val_idx):
        return np.full(len(val_idx), y_train.iloc[train_idx].mean())

    def rule_scores(train_idx, val_idx):
        return features_all["lead_days"].iloc[val_idx].to_numpy(dtype=float)

    folds, _ = cross_validate_scores(constant_scores, y_train, splits)
    run_id, summary = log_cv_run(
        "baseline-constant", {**common, "model": "constant training prevalence", "feature_set": "none"}, folds, tags
    )
    results["baseline-constant"] = {"run_id": run_id, **summary}

    folds, _ = cross_validate_scores(rule_scores, y_train, splits)
    run_id, summary = log_cv_run(
        "baseline-lead-time-rule",
        {**common, "model": "rule: longest lead time in days called first", "feature_set": "lead_days"},
        folds, tags,
    )
    results["baseline-lead-time-rule"] = {"run_id": run_id, **summary}

    simple_pipeline = build_pipeline(LogisticRegression(max_iter=2000), columns=config.SIMPLE_FEATURES, linear=True)
    folds, _ = cross_validate_scores(model_score_fn(simple_pipeline, X_train, y_train), y_train, splits)
    run_id, summary = log_cv_run(
        "baseline-logreg-simple",
        {**common, "model": "LogisticRegression", "C": 1.0, "feature_set": config.SIMPLE_FEATURES},
        folds, tags,
    )
    results["baseline-logreg-simple"] = {"run_id": run_id, **summary}

    candidates = {
        "logreg-tuned": {
            "label": "LogisticRegression",
            "pipeline": build_pipeline(LogisticRegression(max_iter=2000), linear=True),
            "space": {"model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]},
            "kind": "grid",
        },
        "hgb-tuned": {
            "label": "HistGradientBoosting",
            "pipeline": build_pipeline(
                HistGradientBoostingClassifier(early_stopping=False, random_state=config.SEED)
            ),
            "space": {
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_leaf_nodes": [8, 15, 31],
                "model__min_samples_leaf": [20, 50, 100],
                "model__l2_regularization": [0.0, 0.1, 1.0],
                "model__max_iter": [100, 200],
            },
            "kind": "random",
        },
    }

    tuned = {}
    for name, spec in candidates.items():
        best_params, search_path = tune(name, spec["pipeline"], spec["space"], spec["kind"], X_train, y_train, splits)
        pipeline = clone(spec["pipeline"]).set_params(**best_params)
        folds, oof = cross_validate_scores(model_score_fn(pipeline, X_train, y_train), y_train, splits)
        params = {
            **common,
            "model": spec["label"],
            "feature_set": config.NUMERIC_FEATURES,
            "search": spec["kind"],
            **{k.replace("model__", ""): v for k, v in best_params.items()},
        }
        run_id, summary = log_cv_run(name, params, folds, tags, artifacts=[search_path])
        results[name] = {"run_id": run_id, **summary}
        tuned[name] = {"pipeline": pipeline, "oof": oof, "summary": summary, "params": params, "label": spec["label"]}

    summary_table = pd.DataFrame(results).T
    summary_table.index.name = "run"
    summary_table.to_csv(config.REPORTS_DIR / "cv_summary.csv")

    best_name = max(tuned, key=lambda n: tuned[n]["summary"]["cv_pr_auc_mean"])
    best = tuned[best_name]
    threshold = float(np.quantile(best["oof"], 1 - config.FLAG_RATE))

    with mlflow.start_run(run_name="final-model") as run:
        run_id = run.info.run_id
        mlflow.set_tags({**tags, "final_model": "true"})
        mlflow.log_params({
            **{k: str(v) for k, v in best["params"].items()},
            "selected_run": best_name,
            "selection_rule": "highest mean cross-validated PR-AUC",
            "threshold": threshold,
            "threshold_source": "score quantile of out-of-fold predictions at the flag rate",
        })
        mlflow.log_metrics(best["summary"])

        final = clone(best["pipeline"]).fit(X_train, y_train)

        prob = final.predict_proba(X_test)[:, 1]
        model_rank = evaluate.ranking_metrics(y_test, prob)
        model_thr = evaluate.threshold_metrics(y_test, prob, threshold)
        simple_fitted = clone(simple_pipeline).fit(X_train, y_train)
        baselines = {
            "constant": evaluate.ranking_metrics(y_test, np.full(len(y_test), y_train.mean())),
            "rule": evaluate.ranking_metrics(y_test, make_features(X_test)["lead_days"].to_numpy(dtype=float)),
            "simple_logreg": evaluate.ranking_metrics(y_test, simple_fitted.predict_proba(X_test)[:, 1]),
        }
        brier = float(brier_score_loss(y_test, prob))

        n_days = int(test_df["appointment_day"].nunique())
        flagged_per_day = model_rank["flag_rate"] * len(test_df) / n_days
        precision = model_rank["precision_at_flag"]
        break_even = 1.0 / (precision * config.COST_RATIO) if precision > 0 else None
        if break_even is not None and break_even > 1.0:
            break_even = None

        beats_rule = bool(precision > baselines["rule"]["precision_at_flag"])
        lift_ok = bool(model_rank["lift_at_flag"] >= config.MIN_LIFT)
        kill_a = not beats_rule
        kill_b = bool(model_rank["lift_at_flag"] < config.KILL_LIFT)
        if kill_a or kill_b:
            verdict = "DO NOT DEPLOY"
        elif lift_ok:
            verdict = "DEPLOY AS A SUPERVISED PILOT"
        else:
            verdict = "DO NOT DEPLOY YET"
        criteria = {
            "beats_rule": beats_rule, "lift_ok": lift_ok, "kill_a": kill_a, "kill_b": kill_b, "verdict": verdict,
        }

        mlflow.log_metrics({f"test_{k}": v for k, v in model_rank.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in model_thr.items()})
        for base_name, base_metrics in baselines.items():
            mlflow.log_metrics({f"test_baseline_{base_name}_{k}": v for k, v in base_metrics.items()})
        mlflow.log_metrics({
            "test_brier": brier,
            "test_calls_per_day_at_flag_rate": flagged_per_day,
            "criterion_beats_rule": float(beats_rule),
            "criterion_lift_ok": float(lift_ok),
            "kill_a_triggered": float(kill_a),
            "kill_b_triggered": float(kill_b),
        })

        matrix = evaluate.plot_confusion(y_test, prob >= threshold, config.REPORTS_DIR / "confusion_matrix.png")
        importance = permutation_importance(
            final, X_test, y_test, scoring="average_precision", n_repeats=5, random_state=config.SEED, n_jobs=1
        )
        importance_table = pd.DataFrame({
            "feature": X_test.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }).sort_values("importance_mean", ascending=False)
        importance_table.to_csv(config.REPORTS_DIR / "permutation_importance.csv", index=False)
        evaluate.plot_importance(importance_table, config.REPORTS_DIR / "permutation_importance.png")

        slices = evaluate.all_slices(test_df, prob, threshold)
        gaps = evaluate.gap_summary(slices)
        drift = evaluate.psi_table(train_df, test_df)
        slices.to_csv(config.REPORTS_DIR / "slice_analysis.csv", index=False)
        gaps.to_csv(config.REPORTS_DIR / "fairness_gaps.csv", index=False)
        drift.to_csv(config.REPORTS_DIR / "psi_train_vs_test.csv", index=False)

        predictions = pd.DataFrame({
            "appointment_id": test_df["appointment_id"],
            "appointment_day": test_df["appointment_day"],
            "no_show": y_test,
            "probability": prob,
            "flag": (prob >= threshold).astype(int),
        })
        predictions.to_csv(config.REPORTS_DIR / "test_predictions.csv", index=False)

        test_start = str(test_df["appointment_day"].min().date())
        test_end = str(test_df["appointment_day"].max().date())
        test_metrics = {
            "selected_run": best_name,
            "test_period": [test_start, test_end],
            "test_rows": int(len(test_df)),
            "test_days": n_days,
            "model_at_flag_rate": model_rank,
            "model_at_threshold": model_thr,
            "baselines_at_flag_rate": baselines,
            "brier": brier,
            "calls_per_day_at_flag_rate": flagged_per_day,
            "break_even_recovery_rate": break_even,
            "cost_ratio_assumption": config.COST_RATIO,
            "criteria": criteria,
            "confusion_matrix_rows_actual_cols_flagged": matrix.tolist(),
        }
        with open(config.REPORTS_DIR / "test_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(test_metrics, handle, indent=2)
        recommendation = build_recommendation(
            test_start, test_end, model_rank, baselines["rule"], criteria, flagged_per_day, break_even
        )
        with open(config.REPORTS_DIR / "recommendation.md", "w", encoding="utf-8") as handle:
            handle.write(recommendation)

        joblib.dump(final, config.MODEL_PATH)
        reloaded = joblib.load(config.MODEL_PATH)
        reloaded_prob = reloaded.predict_proba(X_test.head(500))[:, 1]
        if not np.allclose(reloaded_prob, prob[:500], atol=1e-12):
            raise RuntimeError("Reloaded model does not reproduce the in-memory predictions")
        mlflow.log_metric("roundtrip_ok", 1.0)

        mlflow.sklearn.log_model(
            final,
            artifact_path="model",
            registered_model_name=config.MODEL_NAME,
            pip_requirements=str(config.ROOT / "requirements.txt"),
        )
        client = MlflowClient()
        versions = client.search_model_versions(f"run_id='{run_id}'")
        version = max(int(v.version) for v in versions)
        client.set_registered_model_alias(config.MODEL_NAME, "final", str(version))

        meta = {
            "model_name": config.MODEL_NAME,
            "model_version": str(version),
            "model_family": best["label"],
            "selected_run": best_name,
            "mlflow_run_id": run_id,
            "mlflow_experiment": config.EXPERIMENT_NAME,
            "training_date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_training_rows": int(len(train_df)),
            "data_md5": md5,
            "git_commit": commit,
            "input_columns": config.RAW_INPUT_COLUMNS,
            "model_features": config.NUMERIC_FEATURES,
            "excluded_inputs": config.DROPPED_INPUTS,
            "threshold": threshold,
            "flag_rate": config.FLAG_RATE,
            "decision_point": f"{config.DECISION_HOUR}:00 on the day before the appointment",
            "prediction_meaning": "Probability that a booked appointment ends in a no-show, scored at 17:00 the day before.",
            "library_versions": library_versions(),
        }
        with open(config.META_PATH, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)

        for path in [
            config.META_PATH,
            config.REPORTS_DIR / "test_metrics.json",
            config.REPORTS_DIR / "recommendation.md",
            config.REPORTS_DIR / "confusion_matrix.png",
            config.REPORTS_DIR / "permutation_importance.png",
            config.REPORTS_DIR / "permutation_importance.csv",
            config.REPORTS_DIR / "slice_analysis.csv",
            config.REPORTS_DIR / "fairness_gaps.csv",
            config.REPORTS_DIR / "psi_train_vs_test.csv",
            config.REPORTS_DIR / "test_predictions.csv",
            config.REPORTS_DIR / "cv_summary.csv",
            config.DATA_PROCESSED_DIR / "cleaning_report.json",
        ]:
            mlflow.log_artifact(str(path))

    print(summary_table[["cv_pr_auc_mean", "cv_pr_auc_std", "cv_precision_at_flag_mean", "cv_precision_at_flag_std", "cv_lift_at_flag_mean"]].round(4).to_string())
    print()
    print(recommendation)
    print(f"Final model run ID: {run_id}")
    print(f"Registered model: {config.MODEL_NAME} version {version} (alias: final)")


if __name__ == "__main__":
    main()
