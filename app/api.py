from contextlib import asynccontextmanager
from datetime import date, datetime, time, timedelta
from typing import Literal

import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, model_validator

from src import config
from src.predict import action_text, load_artifact, score_frame


@asynccontextmanager
async def lifespan(app):
    model, meta = load_artifact()
    app.state.model = model
    app.state.meta = meta
    yield


app = FastAPI(
    title="No-show triage API",
    description="Scores a booked appointment at 17:00 the day before and says whether to call the patient.",
    lifespan=lifespan,
)


class Appointment(BaseModel):
    age: int = Field(ge=0, le=config.AGE_MAX)
    scholarship: Literal[0, 1]
    hipertension: Literal[0, 1]
    diabetes: Literal[0, 1]
    alcoholism: Literal[0, 1]
    scheduled_day: datetime
    appointment_day: date
    prior_appts: int = Field(default=0, ge=0)
    prior_noshows: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def check_consistency(self):
        if self.prior_noshows > self.prior_appts:
            raise ValueError("prior_noshows cannot be larger than prior_appts")
        scheduled = self.scheduled_day.replace(tzinfo=None)
        decision = (
            datetime.combine(self.appointment_day, time.min)
            - timedelta(days=1)
            + timedelta(hours=config.DECISION_HOUR)
        )
        if scheduled > decision:
            raise ValueError(
                f"scheduled_day is after the {config.DECISION_HOUR}:00 decision point on the day before the "
                "appointment, so this appointment is not on the call list"
            )
        return self


class Prediction(BaseModel):
    prediction: int
    probability: float
    threshold: float
    action: str
    model_version: str


@app.get("/health")
def health(request: Request):
    meta = request.app.state.meta
    return {"status": "ok", "model_name": meta["model_name"], "model_version": meta["model_version"]}


@app.post("/predict", response_model=Prediction)
def predict(item: Appointment, request: Request):
    model = request.app.state.model
    meta = request.app.state.meta
    row = pd.DataFrame([{
        "age": item.age,
        "scholarship": item.scholarship,
        "hipertension": item.hipertension,
        "diabetes": item.diabetes,
        "alcoholism": item.alcoholism,
        "scheduled_day": item.scheduled_day.replace(tzinfo=None),
        "appointment_day": datetime.combine(item.appointment_day, time.min),
        "prior_appts": item.prior_appts,
        "prior_noshows": item.prior_noshows,
    }])
    prob, flag = score_frame(model, meta, row)
    return Prediction(
        prediction=int(flag[0]),
        probability=float(prob[0]),
        threshold=float(meta["threshold"]),
        action=action_text(bool(flag[0])),
        model_version=str(meta["model_version"]),
    )
