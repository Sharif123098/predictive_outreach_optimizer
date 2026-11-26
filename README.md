# noshow-triage-mlops

## Recommendation

> Paste the paragraph from `reports/recommendation.md` here after running `python src/train.py`. It is generated from the held-out test result and the charter criteria, so it contains real numbers and a deploy / do-not-deploy verdict. Edit the wording for a head of department, but do not change the verdict unless the charter criteria say so.

Final model run ID: see `mlflow_run_id` in `models/model.json` (also printed at the end of training and tagged `final_model=true` in MLflow; registered as `noshow-triage-model`, alias `final`).

## Problem

At 17:00 on the day before, a clinic front-desk coordinator decides which of tomorrow's patients get a reminder call. The model ranks tomorrow's booked appointments by no-show risk so the limited number of calls goes to the patients most likely to miss. Full decision, owner, success criteria and kill criteria are in [CHARTER.md](CHARTER.md).

The Kaggle no-show dataset is widely used. The question asked of it here is different from the usual "predict no-show and report accuracy": a capacity-limited call list, scored at a fixed decision time, evaluated on a later period, with prior-appointment history built only from outcomes known at that time.

## Data

Source, licence, download date, size and MD5 are recorded in [data/README.md](data/README.md). The raw file is git-ignored; download it from Kaggle into `data/raw/KaggleV2-May-2016.csv`.

## How the point-in-time rules are applied

- **Scored population:** only appointments booked by 17:00 the day before the appointment. Appointments booked later do not exist at decision time and are excluded; counts and their no-show rate are in `data/processed/cleaning_report.json`.
- **Excluded inputs:** `SMS_received` (send time unknown), and `Gender`, `Neighbourhood`, `Handcap` (excluded by charter, used only for slice analysis).
- **History features:** prior appointments and prior no-shows count only appointments dated at least two days before the one being scored.
- **Split:** time-based. Earlier appointment days train, later days test. Tuning uses forward-chaining folds by day inside the training period. The test period is evaluated once, at the end of `src/train.py`.
- **Preprocessing:** every step lives inside one scikit-learn pipeline (`src/features.py`), which the API and UI load from the saved artifact.

## Run it (Windows PowerShell, Python 3.12 or 3.11)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

mkdir data\raw
move KaggleV2-May-2016.csv data\raw\

python src/train.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
python src/evaluate.py
python -m pytest -q

uvicorn app.api:app --reload
streamlit run app/ui.py

docker compose up --build
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` first.

Open the MLflow UI at http://127.0.0.1:5000, the API docs at http://127.0.0.1:8000/docs and the UI at http://localhost:8501. For Docker, run `python src/train.py` first so `models/model.joblib` exists, and commit it.

Example request:

```powershell
$body = '{"age":40,"scholarship":0,"hipertension":1,"diabetes":0,"alcoholism":0,"scheduled_day":"2016-05-02T10:00:00","appointment_day":"2016-05-09","prior_appts":2,"prior_noshows":1}'
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body $body
```

## Tracked runs

`python src/train.py` logs five runs to MLflow, all with parameters, per-fold and mean/std metrics, data MD5 and git commit:

- `baseline-constant`, `baseline-lead-time-rule`, `baseline-logreg-simple`
- `logreg-tuned`, `hgb-tuned` (two model families, tuned inside cross-validation)
- `final-model`: selected by highest mean cross-validated PR-AUC, evaluated once on the test period, registered and aliased `final`

Notebook 02 reads its tables and plots from the tracker, so every reported number traces to a run.

## Structure

```
CHARTER.md  MONITORING.md  README.md  SUBMISSION_CHECKLIST.md
requirements.txt  requirements-dev.txt  Dockerfile  docker-compose.yml  pytest.ini
data/        README.md, raw/ (git-ignored), processed/
notebooks/   01_data_audit_eda, 02_baselines_modelling, 03_evaluation_error_analysis
src/         config, data, features, train, evaluate, predict
models/      model.joblib, model.json
reports/     metrics, slice tables, figures, recommendation.md
app/         api.py (FastAPI), ui.py (Streamlit)
tests/       pytest for features and the API
.github/     CI workflow (tests and Docker build)
```

Layout is my own, separating data, exploration, source code, models and the app.

## Limitations

- The data is one public snapshot from one city. Results may not transfer to another clinic.
- Appointment dates span only a few weeks, so prior-history features are thin early in the period and grow over time. The drift check reflects this.
- The decision point is the calendar day before. For Monday appointments a real clinic would call on Friday.
- The file has no timestamps for `Scholarship`, `Hipertension`, `Diabetes` and `Alcoholism`; they are assumed to be known before booking.
- The cost of a missed slot relative to a call is an assumption in `src/config.py`, not a measured value.

## AI assistance

An AI assistant (Claude) was used to scaffold the code, tests and documentation in this repository. I reviewed and ran all of it and can explain each part.
