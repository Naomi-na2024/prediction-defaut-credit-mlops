"""Tests de l'application et du modèle (lancés automatiquement par GitHub Actions)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import app  # pylint: disable=wrong-import-position

RISKY_CLIENT = {
    "credit_lines_outstanding": 5,
    "loan_amt_outstanding": 1958.93,
    "total_debt_outstanding": 8228.75,
    "income": 26648.44,
    "years_employed": 2,
    "fico_score": 572,
}
SAFE_CLIENT = {
    "credit_lines_outstanding": 0,
    "loan_amt_outstanding": 5221.55,
    "total_debt_outstanding": 3915.47,
    "income": 78039.39,
    "years_employed": 5,
    "fico_score": 605,
}


@pytest.fixture(name="client")
def fixture_client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_home_page_displays_form(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Score FICO" in response.get_data(as_text=True)


def test_api_detects_risky_client(client):
    response = client.post("/api/predict", json=RISKY_CLIENT)
    assert response.status_code == 200
    assert response.get_json()["default"] == 1


def test_api_detects_safe_client(client):
    response = client.post("/api/predict", json=SAFE_CLIENT)
    assert response.status_code == 200
    assert response.get_json()["default"] == 0


def test_form_shows_result(client):
    response = client.post("/", data=RISKY_CLIENT)
    assert "Défaut de paiement probable" in response.get_data(as_text=True)


def test_missing_field_is_rejected(client):
    incomplete = {k: v for k, v in RISKY_CLIENT.items() if k != "income"}
    response = client.post("/api/predict", json=incomplete)
    assert response.status_code == 400


def test_text_value_is_rejected(client):
    response = client.post("/api/predict", json={**RISKY_CLIENT, "income": "abc"})
    assert response.status_code == 400


def test_model_quality_is_kept():
    """Garde-fou : le modèle enregistré doit rester performant."""
    metrics_path = Path(__file__).resolve().parents[1] / "models" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["f1_score"] >= 0.9
