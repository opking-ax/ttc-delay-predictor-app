from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocess import preprocess
from src.train import train

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "bus"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "bus.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "bus"


def run():
    print("=" * 39)
    print("  TTC Bus Delay Predictor - Pipeline")
    print("=" * 39)

    print("\n[1/2] Preprocessing raw bus data ...")
    preprocess(raw_folder=RAW_DIR, output_path=PROCESSED_PATH)

    print("\n[2/2] Training model ...")
    train(processed_csv=PROCESSED_PATH, model_dir=MODEL_DIR, transit_type='bus')

    print("\n[Done] Bus Pipeline complete!")
    print(f"    Model artifact: {MODEL_DIR / 'model.pkl'}")
    print(f"    MLflow UI:      mlflow ui (open http://localhost:5000)")


if __name__ == "__main__":
    run()
