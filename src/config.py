from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw" / "KaggleV2-May-2016.csv"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODEL_PATH = MODELS_DIR / "model.joblib"
META_PATH = MODELS_DIR / "model.json"

TRACKING_DB = ROOT / "mlflow.db"
TRACKING_URI = f"sqlite:///{TRACKING_DB.as_posix()}"
ARTIFACT_URI = (ROOT / "mlruns").as_uri()
EXPERIMENT_NAME = "noshow-triage-mlops"
MODEL_NAME = "noshow-triage-model"

EXPECTED_DATA_MD5 = ""

SEED = 42
DECISION_HOUR = 17
AGE_MIN = 0
AGE_MAX = 110
TEST_SHARE = 0.20
N_SPLITS = 5
N_ITER_HGB = 12

FLAG_RATE = 0.20
MIN_LIFT = 1.25
KILL_LIFT = 1.10
COST_RATIO = 5.0

TARGET = "no_show"

RAW_INPUT_COLUMNS = [
    "age",
    "scholarship",
    "hipertension",
    "diabetes",
    "alcoholism",
    "scheduled_day",
    "appointment_day",
    "prior_appts",
    "prior_noshows",
]

NUMERIC_FEATURES = [
    "age",
    "scholarship",
    "hipertension",
    "diabetes",
    "alcoholism",
    "lead_days",
    "lead_hours",
    "scheduled_hour",
    "scheduled_weekday",
    "appointment_weekday",
    "prior_appts",
    "prior_noshows",
    "prior_noshow_rate",
    "has_history",
]

SIMPLE_FEATURES = ["age", "lead_days", "scholarship"]

SLICE_COLUMNS = ["gender", "age_band", "neighbourhood", "scholarship", "handcap_any", "lead_band"]

DROPPED_INPUTS = {
    "sms_received": "Send time is not in the file, so its existence at 5 pm the day before cannot be shown.",
    "gender": "Excluded as a model input by charter; kept for slice analysis.",
    "neighbourhood": "Excluded as a model input by charter; kept for slice analysis.",
    "handcap": "Disability-related flag; excluded as a model input by charter; kept for slice analysis.",
}

COLUMN_AUDIT = [
    {"column": "PatientId", "written_when": "At patient registration", "available_at_decision_time": "Yes", "role": "Key only; used to build prior-appointment history", "note": "Never a feature itself"},
    {"column": "AppointmentID", "written_when": "At booking", "available_at_decision_time": "Yes", "role": "Row key", "note": "Checked for duplicates"},
    {"column": "Gender", "written_when": "At registration", "available_at_decision_time": "Yes", "role": "Slice analysis only", "note": "Excluded as input by charter"},
    {"column": "ScheduledDay", "written_when": "At booking", "available_at_decision_time": "Only if booked before 17:00 the day before", "role": "Feature (lead time, booking hour, weekday)", "note": "Rows booked later are outside the scored population"},
    {"column": "AppointmentDay", "written_when": "At booking", "available_at_decision_time": "Yes", "role": "Feature (weekday) and split key", "note": "Date only, no clock time in the file"},
    {"column": "Age", "written_when": "At registration or booking", "available_at_decision_time": "Yes", "role": "Feature", "note": "Invalid values removed"},
    {"column": "Neighbourhood", "written_when": "At registration", "available_at_decision_time": "Yes", "role": "Slice analysis only", "note": "Excluded as input by charter"},
    {"column": "Scholarship", "written_when": "Enrolment flag; timestamp not in the file", "available_at_decision_time": "Assumed yes (VERIFY)", "role": "Feature", "note": "Assumption stated in report"},
    {"column": "Hipertension", "written_when": "Patient record flag; timestamp not in the file", "available_at_decision_time": "Assumed yes (VERIFY)", "role": "Feature", "note": "Assumption stated in report"},
    {"column": "Diabetes", "written_when": "Patient record flag; timestamp not in the file", "available_at_decision_time": "Assumed yes (VERIFY)", "role": "Feature", "note": "Assumption stated in report"},
    {"column": "Alcoholism", "written_when": "Patient record flag; timestamp not in the file", "available_at_decision_time": "Assumed yes (VERIFY)", "role": "Feature", "note": "Assumption stated in report"},
    {"column": "Handcap", "written_when": "Patient record flag; timestamp not in the file", "available_at_decision_time": "Assumed yes (VERIFY)", "role": "Slice analysis only", "note": "Excluded as input by charter"},
    {"column": "SMS_received", "written_when": "Relative to the appointment; send time not in the file", "available_at_decision_time": "Cannot be shown", "role": "Excluded", "note": "Possible leakage and confounded with lead time"},
    {"column": "No-show", "written_when": "After the appointment day", "available_at_decision_time": "No", "role": "Target only", "note": "Label"},
]
