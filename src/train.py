from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

TARGET = "is_delayed"

RANDOM_STATE = 42
TEST_SIZE = 0.2

MODELS_REGISTRY = {
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
}


def load_data(path: str | Path) -> pd.DataFrame:
    """Loads the csv file created from the preprocess.py file"""
    return pd.read_csv(path)


def split_features_targets(df: pd.DataFrame, feature_cols: list[str]) -> tuple:
    available = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(available)
    if missing:
        raise ValueError(f"[train] WARNING: Missing features: {missing}")

    X = df[available]
    y = df[TARGET]
    return X, y, available


def get_train_test_split(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Split the X and y features into their train and test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[train] Train: {len(X_train):,} | Test: {len(X_test):,}")

    return X_train, X_test, y_train, y_test


def build_pipeline(model_name: str = "random_forest") -> Pipeline:
    """
    Build a scikit-learn Pipeline that:
        1. Uses 3 different encoders
            - OrdinalEncoder
            - OneHotEncoder
            - TargetEncoder
        2. Trains a multiple different models
            - Random Forest
            - Gradient Bossting
            - Logistic Regression
        3. A passthrough for the remaining features

    Using a Pipeline ensure the same transformations are applied
    identically during training & prediction -- no data leakage.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "ordinal_encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ["time_of_day"],
            ),
            ("target_encoder", TargetEncoder(), ["route", "incident"]),
            ("onehot_encoder", OneHotEncoder(handle_unknown="ignore"), ["direction"]),
            ("passthrough", "passthrough", ['hour', "day_of_week", 'month', 'is_weekend', 'is_am_rush', 'is_pm_rush'])
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", MODELS_REGISTRY[model_name]),       # Using just random forest for now, would add the other at a later time
        ]
    )
    return pipeline


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Compute and return a dict of evaluation metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    )

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted"),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
    }

    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=["on-time", "delayed"])

    print("\n-- Evaluation Metrics --------")
    print(f"\n[train] Confusion Matrix:\n{cm}")
    print(f"\n[train] Classification Report\n{cr}")
    return metrics


def save_pipeline(pipeline, model_dir, model_path):
    """
    Saves the model.pkl to a given output path
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"\n[train] Model saved -> {model_path}")


def train(feature_cols: list[str],processed_csv: Path,model_dir: Path,transit_type: str = "bus",model_name: str = "random_forest",
) -> Pipeline:
    """
    Load processed CSV, train a model, evaluate it, log everything to MLflow, and save
    the artifact.

    Returns the trained Pipeline
    """
    df = load_data(processed_csv)

    X, y, available = split_features_targets(df, feature_cols)

    X_train, X_test, y_train, y_test = get_train_test_split(X, y)

    mlflow.set_experiment(f"ttc-{transit_type}-delay")

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(
            {
                "transit_type": transit_type,
                "model": model_name,
                "n_features": len(available),
                "train_samples": len(X_train),
            }
        )

        pipeline = build_pipeline(model_name)
        pipeline.fit(X_train, y_train)

        metrics = evaluate(pipeline, X_test, y_test)
        for k, v in metrics.items():
            print(f"[train]    {k:10s}: {v:.4f}")
        mlflow.log_metrics(metrics)

        model_path = model_dir / "model.pkl"
        save_pipeline(pipeline, model_dir, model_path)

        mlflow.sklearn.log_model(pipeline, "model")
        mlflow.log_artifact(str(model_path))

    return pipeline
