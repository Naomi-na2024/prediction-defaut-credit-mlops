"""Messages d'erreur de validation, en français."""

from typing import Any

from credit_risk.schemas import LoanApplication

FIELDS = LoanApplication.model_fields


def number(value: float) -> str:
    """10000000.0 -> « 10 000 000 »."""
    return f"{value:,.10g}".replace(",", " ") if abs(value) < 1e15 else f"{value:g}"


def clean_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Erreurs sans la valeur reçue (qui peut être énorme ou non sérialisable)."""
    return [{"type": e["type"], "loc": list(e["loc"]), "msg": e["msg"]} for e in errors]


def french_messages(errors: list[dict[str, Any]]) -> list[str]:
    """Messages d'erreur en français."""
    messages = []
    for error in errors:
        field = next((p for p in reversed(error["loc"]) if p in FIELDS), None)
        label = f"« {FIELDS[field].title} »" if field else "La requête"
        ctx = error.get("ctx", {})
        kind = error["type"]
        if kind == "missing":
            text = "est obligatoire"
        elif kind in ("int_parsing", "int_from_float"):
            text = "doit être un nombre entier"
        elif kind == "float_parsing":
            text = "doit être un nombre"
        elif kind == "greater_than":
            text = f"doit être supérieur à {number(ctx['gt'])}"
        elif kind == "greater_than_equal":
            text = f"doit être supérieur ou égal à {number(ctx['ge'])}"
        elif kind == "less_than_equal":
            text = f"doit être inférieur ou égal à {number(ctx['le'])}"
        elif kind == "finite_number":
            text = "doit être un nombre fini"
        elif kind == "value_error":
            text = "doit être un nombre"
        elif kind == "json_invalid":
            label, text = "Le JSON envoyé", "est invalide"
        elif kind == "extra_forbidden":
            label, text = f"« {error['loc'][-1]} »", "n'est pas un champ attendu"
        else:
            text = error["msg"]
        messages.append(f"{label} {text}.")
    return messages
