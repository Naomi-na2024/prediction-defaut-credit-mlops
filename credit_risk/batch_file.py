"""Évaluation d'un fichier de clients (CSV ou Excel)."""

import io
import re
import tempfile
import time
import uuid
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from credit_risk.messages import french_messages
from credit_risk.model import CreditRiskModel
from credit_risk.schemas import FEATURES, LoanApplication

MAX_BYTES = 20 * 1024 * 1024  # 20 Mo
MAX_ROWS = 100_000
RESULTS_DIR = Path(tempfile.gettempdir()) / "creditscore"
KEEP_SECONDS = 3600  # les résultats sont supprimés au bout d'une heure
TOKEN = re.compile(r"[0-9a-f]{32}")
RESULT_COLUMNS = [
    "score_credit",
    "note",
    "probabilite_defaut",
    "defaut_predit",
    "niveau_risque",
    "alerte",
    "erreur",
]


class FileError(ValueError):
    """Fichier illisible ou mal structuré (message affiché à l'utilisateur)."""


def read_table(filename: str, content: bytes) -> pd.DataFrame:
    name = filename.lower()
    if not name.endswith((".csv", ".txt", ".xlsx")):
        raise FileError("Format non pris en charge : envoyez un fichier .csv ou .xlsx.")
    if len(content) > MAX_BYTES:
        raise FileError("Fichier trop volumineux (20 Mo maximum).")
    if not content.strip():
        raise FileError("Le fichier est vide.")
    try:
        if name.endswith(".xlsx"):
            frame = _read_excel(content)
        else:
            text = _decode(content)
            # index_col=False : une ligne avec trop de colonnes (par exemple une
            # virgule dans un nom) ne doit pas décaler les numéros de ligne
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", pd.errors.ParserWarning)
                frame = pd.read_csv(
                    io.StringIO(text), dtype=str, sep=_separator(text), index_col=False
                )
    except Exception as exc:
        raise FileError(
            "Le fichier n'a pas pu être lu. Vérifiez qu'il s'agit bien d'un CSV "
            "ou d'un fichier Excel (.xlsx)."
        ) from exc

    frame = frame.reset_index(drop=True)  # numérotation 0, 1, 2... = lignes du fichier
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    # pandas renomme une colonne en double « income » en « income.1 »
    names = [re.sub(r"\.\d+$", "", c) for c in frame.columns]
    duplicated = sorted({n for n in names if n in FEATURES and names.count(n) > 1})
    if duplicated:
        raise FileError("Colonne en double : " + ", ".join(duplicated) + ".")
    frame = frame.dropna(how="all")  # lignes entièrement vides
    missing = [f for f in FEATURES if f not in frame.columns]
    if missing:
        raise FileError(
            "Colonnes manquantes : " + ", ".join(missing) + ". "
            "Téléchargez le modèle de fichier pour voir les colonnes attendues."
        )
    if frame.empty:
        raise FileError("Le fichier ne contient aucun client.")
    if len(frame) > MAX_ROWS:
        raise FileError(
            f"Trop de lignes : {MAX_ROWS:,} clients maximum.".replace(",", " ")
        )
    return frame


def _separator(text: str) -> str:
    """« , », « ; » (Excel français) ou tabulation, d'après la ligne d'en-tête."""
    header = text.lstrip("\ufeff").splitlines()[0]
    return max([",", ";", "\t"], key=header.count)


def _read_excel(content: bytes) -> pd.DataFrame:
    """Prend la première feuille qui contient les colonnes attendues."""
    sheets = pd.read_excel(io.BytesIO(content), sheet_name=None, dtype=str)
    for sheet in sheets.values():
        columns = {str(c).strip().lower() for c in sheet.columns}
        if set(FEATURES) <= columns:
            return sheet
    return next(iter(sheets.values()))


def _decode(content: bytes) -> str:
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return content.decode("cp1252")  # CSV enregistré par Excel sous Windows


def score_table(model: CreditRiskModel, frame: pd.DataFrame) -> pd.DataFrame:
    """Ajoute le score à chaque ligne valide et un message d'erreur aux autres."""
    applications, positions = [], []
    errors = [""] * len(frame)
    for i, row in enumerate(frame[FEATURES].to_dict("records")):
        values = {
            k: str(v).strip().replace(",", ".")
            for k, v in row.items()
            if isinstance(v, str) and v.strip()
        }
        try:
            applications.append(LoanApplication.model_validate(values))
            positions.append(i)
        except ValidationError as exc:
            errors[i] = " ".join(french_messages(exc.errors()))

    result = frame.copy()
    for column in RESULT_COLUMNS:
        result[column] = None
    result["erreur"] = errors
    if applications:
        predictions = model.predict(applications)
        rows = result.index[positions]
        result.loc[rows, "score_credit"] = [p["credit_score"] for p in predictions]
        result.loc[rows, "note"] = [p["grade"] for p in predictions]
        result.loc[rows, "probabilite_defaut"] = [p["probability"] for p in predictions]
        result.loc[rows, "defaut_predit"] = [p["default"] for p in predictions]
        result.loc[rows, "niveau_risque"] = [p["risk_level"] for p in predictions]
        result.loc[rows, "alerte"] = [" ; ".join(p["warnings"]) for p in predictions]
    return result


def summarize(result: pd.DataFrame) -> dict[str, Any]:
    valid = result[result["erreur"] == ""]
    grades = Counter(valid["note"])
    total_valid = len(valid)
    # numéro de ligne dans le fichier : +2 pour l'en-tête et la numérotation à 1
    errors = [
        {"line": i + 2, "message": message}
        for i, message in result["erreur"].items()
        if message
    ]
    return {
        "rows": len(result),
        "valid": total_valid,
        "invalid": len(errors),
        "mean_score": round(valid["score_credit"].astype(float).mean())
        if total_valid
        else None,
        "defaults": int(valid["defaut_predit"].astype(int).sum()) if total_valid else 0,
        "flagged": int((valid["alerte"] != "").sum()),
        "grades": [
            {
                "grade": g,
                "count": grades.get(g, 0),
                "share": grades.get(g, 0) / total_valid * 100 if total_valid else 0,
            }
            for g in "ABCD"
        ],
        "errors": errors[:5],
        "preview": [
            {"line": i + 2, **row}
            for i, row in result[
                ["score_credit", "note", "probabilite_defaut", "erreur"]
            ]
            .head(10)
            .to_dict("index")
            .items()
        ],
    }


def save(result: pd.DataFrame) -> str:
    """Enregistre le fichier de résultats et renvoie son identifiant."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    limit = time.time() - KEEP_SECONDS
    for old in RESULTS_DIR.glob("*.csv"):
        if old.stat().st_mtime < limit:
            old.unlink(missing_ok=True)
    token = uuid.uuid4().hex
    # « ; » et virgule décimale : le fichier s'ouvre directement dans Excel français
    output = result.copy()
    for column in [*FEATURES, "probabilite_defaut"]:
        output[column] = output[column].map(_french_number)
    for column in ["score_credit", "defaut_predit"]:  # entiers : 766 et non 766.0
        output[column] = output[column].map(
            lambda v: "" if v is None or v != v else str(int(v))
        )
    output = output.map(_safe_cell)
    output.to_csv(
        RESULTS_DIR / f"{token}.csv", sep=";", index=False, encoding="utf-8-sig"
    )
    return token


def _safe_cell(value: Any) -> Any:
    """Empêche Excel d'exécuter une cellule comme une formule (=, +, -, @).

    Faille connue (« CSV injection ») : un nom de client comme
    =HYPERLINK(...) serait sinon exécuté à l'ouverture du fichier.
    """
    if not isinstance(value, str) or not value or value[0] not in "=+-@\t\r":
        return value
    try:
        float(value.replace(",", "."))  # un nombre négatif reste un nombre
        return value
    except ValueError:
        return "'" + value


def _french_number(value: Any) -> str:
    """1958.93 -> « 1958,93 » ; une valeur qui n'est pas un nombre est gardée."""
    if value is None or (isinstance(value, float) and value != value):
        return ""
    text = str(value).strip()
    try:
        float(text.replace(",", "."))
    except ValueError:
        return text
    return text.replace(".", ",")


def result_path(token: str) -> Path | None:
    if not TOKEN.fullmatch(token):
        return None
    path = RESULTS_DIR / f"{token}.csv"
    return path if path.exists() else None


def template_csv(examples: list[dict[str, Any]]) -> str:
    frame = pd.DataFrame(examples, columns=FEATURES)
    frame.insert(0, "client", [f"client_{i + 1}" for i in range(len(frame))])
    return frame.to_csv(sep=";", decimal=",", index=False)
