from fastapi import FastAPI
from pydantic import BaseModel
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predict import DelayPredictor

app = FastAPI(
    title="TTC Delay Predictor",
    description="Predicts whether a TTC trip will be delayed by more than 15 minutes."
)

prediction = DelayPredictor()

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

@app.post("/predict")
def predict_delay(data: TransportDelay):
    return prediction.predict(data.model_dump())
