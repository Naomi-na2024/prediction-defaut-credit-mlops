"""Tests de l'API et du modèle."""

import io
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from credit_risk.api import EXAMPLES, create_app
from credit_risk.config import MODELS_DIR

RISKY_CLIENT = EXAMPLES["risque"]
SAFE_CLIENT = EXAMPLES["sur"]


@pytest.fixture(name="client", scope="module")
def fixture_client():
    with TestClient(create_app()) as test_client:
        yield test_client


# Supervision


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["X-Request-ID"]


def test_model_info(client):
    info = client.get("/api/v1/model").json()
    assert info["model_name"]
    assert 0 < info["threshold"] < 1
    assert set(info["features"]) == set(RISKY_CLIENT)


def test_openapi_docs_are_available(client):
    assert client.get("/docs").status_code == 200
    assert "/api/v1/predict" in client.get("/openapi.json").json()["paths"]


# Prédiction


def test_api_detects_risky_client(client):
    body = client.post("/api/v1/predict", json=RISKY_CLIENT).json()
    assert body["default"] == 1
    assert body["risk_level"] == "élevé"
    assert body["probability"] >= body["threshold"]


def test_api_detects_safe_client(client):
    body = client.post("/api/v1/predict", json=SAFE_CLIENT).json()
    assert body["default"] == 0
    assert body["risk_level"] == "faible"


def test_credit_score_and_grade(client):
    risky = client.post("/api/v1/predict", json=RISKY_CLIENT).json()
    safe = client.post("/api/v1/predict", json=SAFE_CLIENT).json()
    assert 300 <= risky["credit_score"] < safe["credit_score"] <= 850
    assert (risky["grade"], safe["grade"]) == ("D", "A")


def test_explanation_covers_every_feature(client):
    body = client.post("/api/v1/predict", json=RISKY_CLIENT).json()
    explanation = body["explanation"]
    assert {c["feature"] for c in explanation} == set(RISKY_CLIENT)
    impacts = [abs(c["impact"]) for c in explanation]
    assert impacts == sorted(impacts, reverse=True)


def test_explanation_can_be_disabled(client):
    body = client.post("/api/v1/predict?explain=false", json=RISKY_CLIENT).json()
    assert body["explanation"] is None


def test_batch_prediction(client):
    response = client.post(
        "/api/v1/predict/batch",
        json={"applications": [RISKY_CLIENT, SAFE_CLIENT]},
    )
    assert response.status_code == 200
    assert [p["default"] for p in response.json()["predictions"]] == [1, 0]


def test_more_debt_never_lowers_risk(client):
    """Plus de lignes de crédit => risque plus élevé."""
    probabilities = [
        client.post(
            "/api/v1/predict?explain=false",
            json={**SAFE_CLIENT, "credit_lines_outstanding": lines},
        ).json()["probability"]
        for lines in range(6)
    ]
    assert probabilities == sorted(probabilities)


# Validation des entrées


@pytest.mark.parametrize(
    "payload",
    [
        {k: v for k, v in RISKY_CLIENT.items() if k != "income"},
        {**RISKY_CLIENT, "income": "abc"},
        {**RISKY_CLIENT, "fico_score": 900},
        {**RISKY_CLIENT, "loan_amt_outstanding": -1},
        {**RISKY_CLIENT, "income": 0},
        {**RISKY_CLIENT, "unknown_field": 1},
    ],
    ids=[
        "champ manquant",
        "texte",
        "fico trop haut",
        "montant négatif",
        "revenu nul",
        "champ inconnu",
    ],
)
def test_invalid_input_is_rejected(client, payload):
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422
    assert response.json()["messages"]


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e400", "true"])
def test_special_values_never_crash(client, value):
    """Ces valeurs provoquaient une erreur 500 (ou étaient acceptées pour true)."""
    body = json.dumps({k: v for k, v in RISKY_CLIENT.items() if k != "income"})
    body = body[:-1] + f', "income": {value}}}'
    response = client.post(
        "/api/v1/predict", content=body, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert response.json()["messages"]


def test_request_id_is_sanitized(client):
    assert (
        client.get("/health", headers={"X-Request-ID": "abc-123"}).headers[
            "X-Request-ID"
        ]
        == "abc-123"
    )
    for bad in ["<script>x</script>", "A" * 500]:
        returned = client.get("/health", headers={"X-Request-ID": bad}).headers
        assert returned["X-Request-ID"] != bad


def test_form_accepts_french_decimal_comma(client):
    page = client.post("/", data={**SAFE_CLIENT, "income": "78039,39"}).text
    assert 'data-count="' in page  # un score est affiché


def test_error_messages_are_readable(client):
    page = client.post("/", data={**SAFE_CLIENT, "income": "99999999999"}).text
    assert "inférieur ou égal à 10 000 000" in page


def test_unusual_values_are_flagged(client):
    normal = client.post("/api/v1/predict", json=SAFE_CLIENT).json()
    unusual = client.post(
        "/api/v1/predict", json={**SAFE_CLIENT, "credit_lines_outstanding": 30}
    ).json()
    assert normal["warnings"] == []
    assert "Lignes de crédit en cours" in unusual["warnings"][0]


def test_security_headers(client):
    headers = client.get("/").headers
    assert "default-src 'self'" in headers["Content-Security-Policy"]
    assert headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]


def test_framing_can_be_allowed_for_hugging_face(monkeypatch):
    monkeypatch.setenv("FRAME_ANCESTORS", "https://huggingface.co")
    with TestClient(create_app()) as hf_client:
        headers = hf_client.get("/").headers
    assert "X-Frame-Options" not in headers
    assert (
        "frame-ancestors https://huggingface.co" in headers["Content-Security-Policy"]
    )


def test_empty_batch_is_rejected(client):
    response = client.post("/api/v1/predict/batch", json={"applications": []})
    assert response.status_code == 422


# Interface web


def test_home_page_displays_form(client):
    page = client.get("/").text
    assert "Score FICO" in page
    assert "Situation du client" in page


def test_developers_page(client):
    page = client.get("/developpeurs").text
    assert "Le plus simple : envoyer un fichier" in page
    assert "&#34;credit_score&#34;" in page  # vraie réponse d'exemple


def test_example_prefills_form(client):
    assert 'value="572"' in client.get("/?exemple=risque").text


def test_form_shows_result(client):
    page = client.post("/", data=RISKY_CLIENT).text
    assert "Défaut de paiement probable" in page
    assert "Impact de chaque variable" in page


def test_form_shows_french_errors(client):
    page = client.post("/", data={**SAFE_CLIENT, "fico_score": "100"}).text
    assert "« Score FICO » doit être supérieur ou égal à 300." in page


# Qualité du modèle


def test_model_quality_is_kept():
    """Le modèle doit rester performant."""
    card = json.loads((MODELS_DIR / "model_card.json").read_text(encoding="utf-8"))
    assert card["test_metrics"]["f1_score"] >= 0.95
    assert card["test_metrics"]["roc_auc"] >= 0.98


# Évaluation d'un fichier


def upload(client, filename, content):
    return client.post("/fichier", files={"fichier": (filename, content)})


def test_file_template_can_be_reimported(client):
    template = client.get("/fichier/modele.csv")
    assert template.status_code == 200
    page = upload(client, "modele.csv", template.content).text
    assert "Clients notés" in page
    assert "Lignes en erreur" in page


def test_file_french_csv_with_errors(client):
    content = (
        "nom;credit_lines_outstanding;loan_amt_outstanding;total_debt_outstanding;"
        "income;years_employed;fico_score\n"
        "Alice;0;5221,55;3915,47;78039,39;5;605\n"
        "Bob;5;1958,93;8228,75;26648,44;2;572\n"
        "Erreur;2;3000;9000;abc;3;640\n"
    ).encode("cp1252")
    page = upload(client, "clients.csv", content).text
    assert "Ligne 4 : « Revenu annuel » doit être un nombre." in page
    token = page.split("/fichier/resultats/")[1].split('"')[0]
    result = client.get(f"/fichier/resultats/{token}").content.decode("utf-8-sig")
    lines = result.splitlines()
    assert lines[0].startswith("nom;") and "score_credit;note" in lines[0]
    assert ";A;" in lines[1] and ";D;" in lines[2]
    assert lines[1].startswith("Alice;0;5221,55;")  # virgule française conservée


def test_file_excel(client):
    buffer = io.BytesIO()
    pd.DataFrame([SAFE_CLIENT, RISKY_CLIENT]).to_excel(buffer, index=False)
    page = upload(client, "clients.xlsx", buffer.getvalue()).text
    assert "Télécharger les résultats" in page


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("clients.csv", b"a;b\n1;2\n", "Colonnes manquantes"),
        ("clients.pdf", b"%PDF", "Format non pris en charge"),
        ("clients.xlsx", b"pas un excel", "pas pu être lu"),
    ],
)
def test_file_rejected_with_clear_message(client, filename, content, message):
    assert message in upload(client, filename, content).text


def test_unknown_results_token(client):
    assert client.get("/fichier/resultats/abc").status_code == 404
    assert client.get("/fichier/resultats/" + "0" * 32).status_code == 404


def test_api_file_returns_scored_csv(client):
    template = client.get("/fichier/modele.csv").content
    response = client.post("/api/v1/fichier", files={"fichier": ("c.csv", template)})
    assert response.status_code == 200
    lines = response.content.decode("utf-8-sig").splitlines()
    assert "score_credit" in lines[0]
    assert len(lines) == 4  # en-tête + 3 clients


def test_api_file_bad_file(client):
    files = {"fichier": ("c.csv", b"a;b\n1;2")}
    response = client.post("/api/v1/fichier", files=files)
    assert response.status_code == 422
    assert "Colonnes manquantes" in response.json()["messages"][0]


def test_formula_injection_is_neutralized(client):
    header = "nom," + ",".join(SAFE_CLIENT)
    row = ",".join(str(v) for v in SAFE_CLIENT.values())
    content = f'{header}\n"=HYPERLINK(""http://x"")",{row}\n'.encode()
    response = client.post("/api/v1/fichier", files={"fichier": ("c.csv", content)})
    result = pd.read_csv(io.BytesIO(response.content), sep=";", encoding="utf-8-sig")
    assert result["nom"][0].startswith("'=")  # Excel l'affichera comme du texte


def test_excel_second_sheet_and_duplicate_column(client):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        pd.DataFrame({"info": ["accueil"]}).to_excel(
            writer, sheet_name="Accueil", index=False
        )
        pd.DataFrame([SAFE_CLIENT]).to_excel(writer, sheet_name="Clients", index=False)
    assert "Clients notés" in upload(client, "c.xlsx", buffer.getvalue()).text

    values = ",".join(str(v) for v in SAFE_CLIENT.values())
    duplicated = f"income,{','.join(SAFE_CLIENT)}\n1,{values}\n".encode()
    assert "Colonne en double : income" in upload(client, "c.csv", duplicated).text


def test_row_with_extra_columns_does_not_crash(client):
    """Une virgule non protégée dans un nom provoquait une erreur 500."""
    header = "nom," + ",".join(SAFE_CLIENT)
    row = ",".join(str(v) for v in SAFE_CLIENT.values())
    content = f'{header}\n=HYPERLINK("http://x","Clic"),{row}\nAlice,{row}\n'.encode()
    response = upload(client, "c.csv", content)
    assert response.status_code == 200
    assert "Ligne 2 :" in response.text  # la ligne cassée est signalée
    assert "Clients notés" in response.text


def test_html_in_filename_is_escaped(client):
    template = client.get("/fichier/modele.csv").content
    page = upload(client, "<script>alert(1)</script>.csv", template).text
    assert "<script>alert(1)" not in page
