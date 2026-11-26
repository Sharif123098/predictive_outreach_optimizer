import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from src import config
from src.predict import action_text, load_artifact, score_frame


@st.cache_resource
def get_artifact():
    return load_artifact()


st.set_page_config(page_title="No-show triage", layout="centered")
st.title("Next-day reminder-call triage")
st.write(
    "Enter one booked appointment as it looks at 17:00 the day before. "
    "The model estimates the chance the patient will not attend and says whether to call."
)

model, meta = get_artifact()

with st.form("appointment"):
    age = st.number_input("Age", min_value=0, max_value=config.AGE_MAX, value=35, step=1)
    left, right = st.columns(2)
    scholarship = left.checkbox("Scholarship")
    hipertension = right.checkbox("Hypertension")
    diabetes = left.checkbox("Diabetes")
    alcoholism = right.checkbox("Alcoholism")
    appointment_date = st.date_input("Appointment date", value=date.today() + timedelta(days=1))
    scheduled_date = st.date_input("Date the appointment was booked", value=date.today() - timedelta(days=7))
    scheduled_time = st.time_input("Time the appointment was booked", value=time(10, 0))
    prior_appts = st.number_input("Earlier appointments of this patient", min_value=0, value=0, step=1)
    prior_noshows = st.number_input("Earlier no-shows of this patient", min_value=0, value=0, step=1)
    submitted = st.form_submit_button("Score appointment")

if submitted:
    scheduled = datetime.combine(scheduled_date, scheduled_time)
    decision = (
        datetime.combine(appointment_date, time.min)
        - timedelta(days=1)
        + timedelta(hours=config.DECISION_HOUR)
    )
    if prior_noshows > prior_appts:
        st.error("Earlier no-shows cannot be larger than earlier appointments.")
    elif scheduled > decision:
        st.error(
            f"This appointment was booked after {config.DECISION_HOUR}:00 on the day before, "
            "so it is not on the call list."
        )
    else:
        row = pd.DataFrame([{
            "age": int(age),
            "scholarship": int(scholarship),
            "hipertension": int(hipertension),
            "diabetes": int(diabetes),
            "alcoholism": int(alcoholism),
            "scheduled_day": scheduled,
            "appointment_day": datetime.combine(appointment_date, time.min),
            "prior_appts": int(prior_appts),
            "prior_noshows": int(prior_noshows),
        }])
        prob, flag = score_frame(model, meta, row)
        st.metric("Chance of no-show", f"{prob[0]:.1%}")
        st.caption(f"Call threshold: {meta['threshold']:.1%}. Model version {meta['model_version']}.")
        if flag[0]:
            st.warning(action_text(True))
        else:
            st.success(action_text(False))
