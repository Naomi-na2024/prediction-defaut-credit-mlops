"""Configuration (variables d'environnement)."""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "loan_data.csv"


@dataclass(frozen=True)
class Settings:
    model_path: Path
    model_card_path: Path
    decision_threshold: float | None
    log_level: str
    # sites autorisés à afficher l'application dans une page (ex. Hugging Face)
    frame_ancestors: str = "'none'"

    @classmethod
    def from_env(cls) -> "Settings":
        threshold = os.getenv("DECISION_THRESHOLD")
        return cls(
            model_path=Path(os.getenv("MODEL_PATH", MODELS_DIR / "model.joblib")),
            model_card_path=Path(
                os.getenv("MODEL_CARD_PATH", MODELS_DIR / "model_card.json")
            ),
            # par défaut : seuil choisi à l'entraînement
            decision_threshold=float(threshold) if threshold else None,
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            frame_ancestors=os.getenv("FRAME_ANCESTORS", "'none'"),
        )
