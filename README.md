# TTC Delay Predictor

Predicts whether a Toronto Transit Commission (TTC) bus, streetcar and/or subway trip will be delayed by more than 5 minutes, based on route, time of day and incident type.

Built to demonstrate an end-to-end ML Pipeline:
**data -> preprocessing -> training -> API -> deployed demo**

---

## Architecture

```
ttc-delay-app/
├── data/
│   ├── raw/              # TTC Open Data CSVs (not committed)
│   └── processed/        # Cleaned, feature-engineered CSVs
├── models/
│   ├── bus/
│   │   ├── random_forest.pkl
│   │   ├── gradient_boosting.pkl
│   │   └── logistic_regression.pkl
│   ├── streetcar/
│   │   ├── random_forest.pkl
│   │   ├── gradient_boosting.pkl
│   │   └── logistic_regression.pkl
│   └── subway/
│   │   ├── random_forest.pkl
│   │   ├── gradient_boosting.pkl
│   │   └── logistic_regression.pkl
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── pipelines/
│       ├── bus.py
│       ├── streetcar.py
│       └── subway.py
├── api/
│   └── main.py           # FastAPI REST endpoint
├── app/
│   └── gradio_app.py     # Gradio demo UI
├── Dockerfile
├── Dockerfile.gradio
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download data
Get TTC transport Delay Data (2022-2024) from
[Toronto Open Data Portal](https://open.toronto.ca/)

| Transit Type | Dataset | Folder |
| --- |---|---|
| Bus | [TTC Bus Delay](https://open.toronto.ca/dataset/ttc-bus-delay-data/) | `data/raw/bus/`
| Streetcar | [TTC Subway Delay](https://open.toronto.ca/dataset/ttc-subway-delay-data/) | `data/raw/streetcar/`
| Subway | [TTC Streetcar Delay](https://open.toronto.ca/dataset/ttc-streetcar-delay-data/) | `data/raw/subway/`

### 3. Train the model
```bash
python -m src.pipelines.<transport>
```
This will
- Preprocess and filter 2022-2024 data
- Engineer temporal features (hour, rush hour flags, etc.)
- Train a RandomForest Classifier
- Log the experiment to MLFlow
- Save model artifacts to `models/<transport>/`
    - where transport is either bus, streetcar or subway

### 4. View MLflow experiments
```bash
mlflow ui
# Open http://localhost:5000
```

### 5. Run the API
```bash
uvicorn api.main:app --reload
# Docs at https://localhost:8000/docs
```

### 6. Run the Gradio demo
```bash
python app/gradio_app.
# Open at http://127.0.0.1:7860
```

### 7. Run tests
```bash
pytest tests/ -v
```

---

## API Usage

```bash
curl -X POST http://localhost:8000/predict/<transport> \
  -H "Content-Type: application/json" \
  -d '{
    "route": 29,
    "hour": 8,
    "day_of_week": 0,
    "month": 3,
    "is_weekend": 0,
    "is_am_rush": 1,
    "is_pm_rush": 0,
    "incident": "mechanical",
    "direction": "south"
  }'
```
Response:
```json
{
  "prediction": 1,
  "label": "delayed",
  "delay_probability": 0.7341
}
```

---

## Docker

```bash
# Build
docker build -t ttc-delay-app .

# Run
docker run -p 8000:8000 ttc-delay-app
```

---

## Transit-Specific Notes
- **Bus & Streetcar:** share the same feature set. Streetcar raw columns (`line`, `bound`) are normalized to the shared schema (`route`, `direction`) before preprocessing.
- **Subway:** uses delay codes (e.g. `SUDP`, `MUPR`) instead of plain English incident names. Includes `location` (station name) as an additional feature. Raw columns (`code`, `bound`, `station`, `line`) are normalized before preprocessing.
---

## Known Limitations
- Random split used for train/test: A time-based split would better simulate real deployment conditions
- Subway incident codes are used as-is without mapping to plain English categories
- Pipeline currently designed for offline batch training — not real-time retraining
- Streetcar and subway Gradio inputs use the same incident list as bus. Transit-specific incident lists would improve UX
---

## Data Source
City of Toronto Open Data Portal. Data filtered to 2022–2024 to reflect post-COVID operations.

- [TTC Bus Delay Data](https://open.toronto.ca/dataset/ttc-bus-delay-data/)
- [TTC Subway Delay Data](https://open.toronto.ca/dataset/ttc-subway-delay-data/)
- [TTC Streetcar Delay Data](https://open.toronto.ca/dataset/ttc-streetcar-delay-data/)
