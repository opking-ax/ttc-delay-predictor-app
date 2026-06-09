from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predict import DelayPredictor

app = FastAPI(
    title="TTC Delay Predictor",
    description="Predicts whether a TTC trip will be delayed by more than 15 minutes."
)

TRANSIT_TYPES = ["bus", "streetcar", "subway"]
MODEL_NAMES   = ["random_forest", "gradient_boosting",
                 "logistic_regression"]

PREDICTORS = {
    (t, m): DelayPredictor(t, m)
    for t in TRANSIT_TYPES
    for m in MODEL_NAMES
}

class TransportDelay(BaseModel):
    hour: int
    route: str
    incident: str
    direction: str
    day_of_week: int
    month: int
    time_of_day: str
    is_weekend: int
    is_am_rush: int
    is_pm_rush: int

@app.post("/predict/{transit_type}/{model_name}")
def predict_delay(transit_type: str, model_name: str, data: TransportDelay):
    key = (transit_type, model_name)
    if key not in PREDICTORS:
        raise HTTPException(status_code=400, detail=f"Invalid combination: {transit_type} / {model_name}")
    return PREDICTORS[key].predict(data.model_dump())


@app.get("/health")
def health():
    return {"status": "ok"}
