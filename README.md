# TTC Delay Predictor
Predicts whether a Toronto Transit Commission (TTC) bus trip will be delayed by more than 15 minutes, based on route, time of day and incident type and more.

Built to demonstrate an end-to-end ML Pipeline:
**data -> preprocessing -> training -> API -> deployed demo**

---

## Architecture

```
ttc-delay-predictor-app/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── bus/
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── pipelines/
│       └──bus.py
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
- [TTC Bus Delay](https://open.toronto.ca/dataset/ttc-bus-delay-data/)
and place them in `data/raw/bus/`.

### 3. Train the model
```bash
python -m src.pipelines.bus
```
This will
- Preprocess and filter 2022-2024 data
- Engineer temporal features (hour, rush hour flags, etc.)
- Train a RandomForest Classifier
- Log the experiment to MLFlow
- Save model artifacts to `models/bus/`

---

## Data Source

[TTC Open Data — Bus Delay Data](https://open.toronto.ca/dataset/ttc-bus-delay-data/)
City of Toronto Open Data Portal. Data filtered to 2022–2024 to reflect post-COVID operations.
