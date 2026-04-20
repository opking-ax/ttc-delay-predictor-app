from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DelayPredictor:
    def __init__(self, transit_type: str = "bus"):
        self.transit_type = transit_type
        model_path = PROJECT_ROOT / "models" / transit_type / "model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"[predict] No trained model found at {model_path}"
                f" Run the {transit_type} pipeline first:\n"
                f"   python -m src.pipelines.{transit_type}"
            )
        self.pipeline = joblib.load(model_path)
        print(f"[predict] Loaded {transit_type} model from {model_path}")

    def __build_input(self, raw_dict: dict[str, str | int]) -> pd.DataFrame:
        """Package raw inputs into the DataFrame format the pipeline expects"""
        return pd.DataFrame(
            [
                {
                    "hour": int(raw_dict["hour"]),
                    "route": str(raw_dict["route"]),
                    "incident": str(raw_dict["incident"]),
                    "direction": str(raw_dict["direction"]),
                    "day_of_week": int(raw_dict["day_of_week"]),
                    "month": int(raw_dict["month"]),
                    "time_of_day": str(raw_dict["time_of_day"]),
                    "is_weekend": int(raw_dict["is_weekend"]),
                    "is_am_rush": int(raw_dict["is_am_rush"]),
                    "is_pm_rush": int(raw_dict["is_pm_rush"]),
                }
            ]
        )

    def predict(self, raw_dict: dict[str, str | int]) -> dict[str, bool | float | str]:
        """
        Predict the data and return a dict as: {is_delayed, probability, label}
        """
        X = self.__build_input(raw_dict)

        prob = float(self.pipeline.predict_proba(X)[0, 1])
        is_delayed = bool(self.pipeline.predict(X)[0])
        label = "Delayed > 15 min" if is_delayed else "On Time / Minor Delay"

        return {"is_delayed": is_delayed, "probability": round(prob, 4), "label": label}
