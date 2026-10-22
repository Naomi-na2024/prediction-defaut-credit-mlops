"""Schémas des données de l'API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LoanApplication(BaseModel):
    """Situation financière d'un client."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        json_schema_extra={
            "examples": [
                {
                    "credit_lines_outstanding": 5,
                    "loan_amt_outstanding": 1958.93,
                    "total_debt_outstanding": 8228.75,
                    "income": 26648.44,
                    "years_employed": 2,
                    "fico_score": 572,
                }
            ]
        },
    )

    credit_lines_outstanding: int = Field(
        ge=0, le=50, title="Lignes de crédit en cours", json_schema_extra={"unit": ""}
    )
    loan_amt_outstanding: float = Field(
        ge=0,
        le=1_000_000,
        title="Montant du prêt restant dû",
        json_schema_extra={"unit": "$"},
    )
    total_debt_outstanding: float = Field(
        ge=0, le=1_000_000, title="Dette totale", json_schema_extra={"unit": "$"}
    )
    income: float = Field(
        gt=0, le=10_000_000, title="Revenu annuel", json_schema_extra={"unit": "$"}
    )
    years_employed: int = Field(
        ge=0,
        le=60,
        title="Ancienneté professionnelle",
        json_schema_extra={"unit": "ans"},
    )
    fico_score: int = Field(
        ge=300, le=850, title="Score FICO", json_schema_extra={"unit": "/ 850"}
    )

    @field_validator("*", mode="before")
    @classmethod
    def no_boolean(cls, value):
        # sinon true / false seraient acceptés comme 1 / 0
        if isinstance(value, bool):
            raise ValueError("doit être un nombre")
        return value


FEATURES: list[str] = list(LoanApplication.model_fields)

RiskLevel = Literal["faible", "modéré", "élevé"]


class Contribution(BaseModel):
    feature: str
    label: str
    value: float
    impact: float = Field(
        description="Effet sur la log-cote du défaut, par rapport à un client "
        "médian. Positif = augmente le risque."
    )
    points: int = Field(description="Effet en points de score (négatif = baisse)")


class Prediction(BaseModel):
    default: int = Field(description="1 si un défaut de paiement est prédit")
    probability: float = Field(ge=0, le=1, description="Probabilité de défaut")
    risk_level: RiskLevel
    credit_score: int = Field(ge=300, le=850, description="Score de crédit")
    grade: Literal["A", "B", "C", "D"]
    threshold: float
    warnings: list[str] = Field(
        default=[], description="Valeurs en dehors de celles vues à l'entraînement"
    )
    explanation: list[Contribution] | None = None


class BatchRequest(BaseModel):
    applications: list[LoanApplication] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    model_version: str
    predictions: list[Prediction]


class PredictionResponse(Prediction):
    model_version: str
