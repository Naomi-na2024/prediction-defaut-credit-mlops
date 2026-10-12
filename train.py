"""Entraîne le modèle de prédiction du défaut de paiement et l'enregistre.

Le prétraitement (standardisation) est inclus dans un Pipeline scikit-learn :
l'application reçoit donc les valeurs brutes saisies par l'utilisateur,
exactement comme pendant l'entraînement.

Utilisation :
    python train.py            # entraîne et enregistre models/model.pkl
    python train.py --mlflow   # enregistre en plus le run dans MLflow
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "loan_data.csv"
MODEL_PATH = ROOT / "models" / "model.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"

FEATURES = [
    "credit_lines_outstanding",
    "loan_amt_outstanding",
    "total_debt_outstanding",
    "income",
    "years_employed",
    "fico_score",
]
TARGET = "default"

PARAMS = {
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42,
}


def build_pipeline() -> Pipeline:
    """Prétraitement + modèle, regroupés pour éviter tout décalage en production."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", DecisionTreeClassifier(**PARAMS)),
        ]
    )


def main(use_mlflow: bool = False) -> dict:
    data = pd.read_csv(DATA_PATH)
    x, y = data[FEATURES], data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_prob = pipeline.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if use_mlflow:
        import mlflow  # pylint: disable=import-outside-toplevel,import-error
        import mlflow.sklearn  # pylint: disable=import-outside-toplevel,import-error

        mlflow.set_experiment("prediction-defaut-credit")
        with mlflow.start_run(run_name="decision_tree"):
            mlflow.log_params(PARAMS)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, "model", input_example=x_test.head(1))

    print(f"Modèle enregistré dans {MODEL_PATH}")
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mlflow", action="store_true", help="Enregistrer le run dans MLflow"
    )
    main(use_mlflow=parser.parse_args().mlflow)
