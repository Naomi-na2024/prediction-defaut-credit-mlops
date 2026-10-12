"""Application web Flask : estime le risque de défaut de paiement d'un client."""

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

MODEL_PATH = Path(__file__).parent / "models" / "model.pkl"

# Champs du formulaire : (nom technique, libellé, type, minimum, unité)
FIELDS = [
    ("credit_lines_outstanding", "Lignes de crédit en cours", int, 0, ""),
    ("loan_amt_outstanding", "Montant du prêt restant dû", float, 0, "$"),
    ("total_debt_outstanding", "Dette totale", float, 0, "$"),
    ("income", "Revenu annuel", float, 0, "$"),
    ("years_employed", "Ancienneté professionnelle", int, 0, "ans"),
    ("fico_score", "Score FICO", int, 300, "/ 850"),
]

app = Flask(__name__)
model = joblib.load(MODEL_PATH)


def parse_features(source) -> dict:
    """Lit et valide les 6 variables. Lève ValueError si une valeur est invalide."""
    features = {}
    for name, label, cast, minimum, _unit in FIELDS:
        raw = source.get(name)
        if raw in (None, ""):
            raise ValueError(f"Le champ « {label} » est obligatoire.")
        try:
            value = cast(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Le champ « {label} » doit être un nombre.") from exc
        if value < minimum:
            raise ValueError(f"Le champ « {label} » doit être au moins {minimum}.")
        features[name] = value
    return features


def predict(features: dict) -> dict:
    """Renvoie la classe prédite et la probabilité de défaut."""
    frame = pd.DataFrame([features])
    probability = float(model.predict_proba(frame)[0, 1])
    return {"default": int(probability >= 0.5), "probability": round(probability, 4)}


@app.route("/", methods=["GET", "POST"])
def home():
    result, error, values = None, None, {}
    if request.method == "POST":
        values = request.form.to_dict()
        try:
            result = predict(parse_features(request.form))
        except ValueError as exc:
            error = str(exc)
    return render_template(
        "index.html", fields=FIELDS, values=values, result=result, error=error
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Point d'accès JSON, pratique pour intégrer le modèle à un autre service."""
    try:
        features = parse_features(request.get_json(silent=True) or {})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(predict(features))


@app.route("/health")
def health():
    """Vérification utilisée par Docker / AWS pour savoir si le service répond."""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
