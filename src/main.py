import pandas as pd
from pipelines.bus import run as bus_run
from pipelines.subway import run as subway_run
from pipelines.streetcar import run as streetcar_run


def run_pipelines():

    print("\n=== Transit Pipeline Status ===\n")

    print("[Bus] Running...")
    bus_run()

    print("[Streetcar] Running...")
    streetcar_run()

    print("[Subway] Running...")
    subway_run()

    print("\n=== Services ===")
    print("MLflow UI: mlflow ui (http://localhost:5000)\n")


if __name__ == "__main__":
    run_pipelines()
