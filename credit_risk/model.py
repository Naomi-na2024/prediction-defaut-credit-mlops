"""Chargement du modèle et prédictions."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from credit_risk.schemas import FEATURES, LoanApplication

# Score de crédit (méthode scorecard) : 520 points quand le client a autant de
# chances de rembourser que de faire défaut, et 7 points de plus à chaque fois
# que cette cote double (échelle adaptée à la dispersion de ces données)
SCORE_BASE, ODDS_BASE, PDO = 520, 1, 7
SCORE_MIN, SCORE_MAX = 300, 850
POINTS_PER_LOG_ODDS = PDO / math.log(2)
GRADES = [(700, "A"), (620, "B"), (540, "C"), (SCORE_MIN, "D")]
RISK_LEVELS = {"A": "faible", "B": "faible", "C": "modéré", "D": "élevé"}


def credit_score(log_odds_default: float) -> int:
    score = SCORE_BASE + POINTS_PER_LOG_ODDS * (-log_odds_default - math.log(ODDS_BASE))
    return round(min(max(score, SCORE_MIN), SCORE_MAX))


def grade(score: int) -> str:
    return next(letter for limit, letter in GRADES if score >= limit)


@dataclass
class CreditRiskModel:
    pipeline: Pipeline
    card: dict[str, Any]
    threshold: float

    @classmethod
    def load(
        cls, model_path: Path, card_path: Path, threshold: float | None = None
    ) -> "CreditRiskModel":
        pipeline = joblib.load(model_path)
        card = json.loads(Path(card_path).read_text(encoding="utf-8"))
        return cls(
            pipeline, card, threshold if threshold is not None else card["threshold"]
        )

    @property
    def version(self) -> str:
        return self.card["model_version"]

    def _probabilities(self, frame: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(frame[FEATURES])[:, 1]

    def _log_odds(self, frame: pd.DataFrame) -> np.ndarray:
        """Log-cote du défaut (plus précis que la probabilité près de 0 ou 1)."""
        if hasattr(self.pipeline, "decision_function"):
            return self.pipeline.decision_function(frame[FEATURES])
        p = np.clip(self._probabilities(frame), 1e-12, 1 - 1e-12)
        return np.log(p / (1 - p))

    def out_of_range(self, application: LoanApplication) -> list[str]:
        """Variables en dehors de ce que le modèle a vu à l'entraînement."""
        fields = LoanApplication.model_fields
        warnings = []
        for feature, (low, high) in self.card.get("feature_ranges", {}).items():
            value = getattr(application, feature)
            if not low <= value <= high:
                warnings.append(
                    f"{fields[feature].title} : {value:g} "
                    f"(valeurs vues entre {low:g} et {high:g})"
                )
        return warnings

    def predict(
        self, applications: list[LoanApplication], explain: bool = False
    ) -> list[dict[str, Any]]:
        frame = pd.DataFrame([app.model_dump() for app in applications])
        probabilities = self._probabilities(frame)
        log_odds = self._log_odds(frame)
        results = []
        for app, probability, lo in zip(
            applications, probabilities, log_odds, strict=True
        ):
            score = credit_score(float(lo))
            result = {
                "default": int(probability >= self.threshold),
                "probability": round(float(probability), 4),
                "risk_level": RISK_LEVELS[grade(score)],
                "credit_score": score,
                "grade": grade(score),
                "threshold": self.threshold,
                "warnings": self.out_of_range(app),
            }
            if explain:
                result["explanation"] = self.explain(app, float(lo))
            results.append(result)
        return results

    def explain(
        self, application: LoanApplication, log_odds: float
    ) -> list[dict[str, Any]]:
        """Effet de chaque variable : on la remplace par la médiane
        et on regarde comment la log-cote du défaut change."""
        values = application.model_dump()
        reference = self.card["reference_values"]
        variants = pd.DataFrame([values] * len(FEATURES))
        for i, feature in enumerate(FEATURES):
            variants.loc[i, feature] = reference[feature]
        counterfactual = self._log_odds(variants)

        fields = LoanApplication.model_fields
        contributions = []
        for feature, lo_ref in zip(FEATURES, counterfactual, strict=True):
            impact = log_odds - float(lo_ref)
            contributions.append(
                {
                    "feature": feature,
                    "label": fields[feature].title,
                    "value": values[feature],
                    "impact": round(impact, 3) + 0.0,
                    "points": round(-POINTS_PER_LOG_ODDS * impact) + 0,
                }
            )
        return sorted(contributions, key=lambda c: abs(c["impact"]), reverse=True)
