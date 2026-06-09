from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocess import preprocess
from src.train import train

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "bus"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "bus.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "bus"
MODELS = ["random_forest", "gradient_boosting", "logistic_regression"]


def run():
    print("=" * 39)
    print("  TTC Bus Delay Predictor - Pipeline")
    print("=" * 39)

    print("\n[1/2] Preprocessing raw bus data ...")
    preprocess(
        raw_folder=RAW_DIR,
        output_path=PROCESSED_PATH,
        transit_type="bus",
        column_maps={},
        extra_drops=["location"],
    )

    print("\n[2/2] Training model ...")
    features_cols = [
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "is_am_rush",
        "is_pm_rush",
        "direction",
        "time_of_day",
        "route",
        "incident",
    ]
    for model_name in MODELS:
        train(features_cols, PROCESSED_PATH, MODEL_DIR, "bus", model_name=model_name)
        print("\n[Done] Bus Pipeline complete!")
        print(f"    Model artifact: {MODEL_DIR / f'{model_name}.pkl'}")
