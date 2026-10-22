"""Entraînement du modèle.

python -m credit_risk.train            # entraîne et sauvegarde le modèle
python -m credit_risk.train --mlflow   # + suivi dans MLflow
"""

import argparse
import json
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from credit_risk.config import DATA_PATH, MODELS_DIR
from credit_risk.schemas import FEATURES

TARGET = "default"
SEED = 42
CV_SCORING = ["roc_auc", "average_precision", "f1"]


def candidates() -> dict[str, Pipeline]:
    """Modèles à comparer."""
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=SEED)),
            ]
        ),
        "decision_tree": Pipeline(
            [
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=6, min_samples_leaf=20, random_state=SEED
                    ),
                )
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        min_samples_leaf=5,
                        n_jobs=-1,
                        random_state=SEED,
                    ),
                )
            ]
        ),
        "gradient_boosting": Pipeline(
            [("model", HistGradientBoostingClassifier(random_state=SEED))]
        ),
    }


def best_threshold(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """Seuil qui maximise le F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return round(float(thresholds[np.argmax(f1[:-1])]), 3)


def evaluate(y_true, y_prob, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "average_precision": round(average_precision_score(y_true, y_prob), 4),
        "brier_score": round(brier_score_loss(y_true, y_prob), 4),
    }


def main(use_mlflow: bool = False) -> dict:
    data = pd.read_csv(DATA_PATH)
    x, y = data[FEATURES], data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=SEED, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    # comparaison des modèles (validation croisée)
    comparison = {}
    for name, pipeline in candidates().items():
        scores = cross_validate(pipeline, x_train, y_train, cv=cv, scoring=CV_SCORING)
        comparison[name] = {
            metric: round(float(scores[f"test_{metric}"].mean()), 4)
            for metric in CV_SCORING
        }
        print(f"{name:<22} {comparison[name]}")

    best_name = max(
        comparison,
        key=lambda n: (comparison[n]["average_precision"], comparison[n]["roc_auc"]),
    )
    pipeline = candidates()[best_name]

    # choix du seuil, sans utiliser le jeu de test
    oof_prob = cross_val_predict(
        pipeline, x_train, y_train, cv=cv, method="predict_proba"
    )[:, 1]
    threshold = best_threshold(y_train, oof_prob)

    # entraînement final et évaluation
    pipeline.fit(x_train, y_train)
    y_prob = pipeline.predict_proba(x_test)[:, 1]
    metrics = evaluate(y_test, y_prob, threshold)
    importance = permutation_importance(
        pipeline, x_test, y_test, scoring="roc_auc", n_repeats=5, random_state=SEED
    )

    now = datetime.now(UTC)
    card = {
        "model_name": best_name,
        "model_version": now.strftime("%Y.%m.%d-%H%M"),
        "trained_at": now.isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "threshold": threshold,
        "dataset": {
            "rows": len(data),
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "default_rate": round(float(y.mean()), 4),
        },
        "features": FEATURES,
        "reference_values": {f: float(x_train[f].median()) for f in FEATURES},
        "feature_ranges": {
            f: [float(x_train[f].min()), float(x_train[f].max())] for f in FEATURES
        },
        "cross_validation": comparison,
        "test_metrics": metrics,
        "confusion_matrix": confusion_matrix(
            y_test, (y_prob >= threshold).astype(int)
        ).tolist(),
        "feature_importance": {
            f: round(float(v), 4)
            for f, v in sorted(
                zip(FEATURES, importance.importances_mean, strict=True),
                key=lambda t: -t[1],
            )
        },
    }

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / "model.joblib", compress=3)
    (MODELS_DIR / "model_card.json").write_text(
        json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if use_mlflow:
        log_to_mlflow(card, comparison, pipeline, x_test)

    print(f"\nModèle retenu : {best_name} (seuil {threshold})")
    print(json.dumps(metrics, indent=2))
    return card


def log_to_mlflow(card, comparison, pipeline, x_test) -> None:
    import mlflow
    import mlflow.sklearn

    mlflow.set_experiment("prediction-defaut-credit")
    with mlflow.start_run(run_name=f"train-{card['model_version']}"):
        for name, scores in comparison.items():
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_param("model", name)
                mlflow.log_metrics({f"cv_{k}": v for k, v in scores.items()})
        mlflow.log_params({"model": card["model_name"], "threshold": card["threshold"]})
        mlflow.log_metrics(card["test_metrics"])
        mlflow.log_dict(card, "model_card.json")
        mlflow.sklearn.log_model(
            pipeline,
            "model",
            input_example=x_test.head(1),
            registered_model_name="prediction-defaut-credit",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mlflow", action="store_true", help="Enregistrer les runs dans MLflow"
    )
    main(use_mlflow=parser.parse_args().mlflow)
