"""API et interface web."""

import hashlib
import json
import logging
import math
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import ValidationError

from credit_risk import __version__, batch_file
from credit_risk.config import Settings
from credit_risk.messages import clean_errors, french_messages
from credit_risk.model import SCORE_MAX, SCORE_MIN, CreditRiskModel
from credit_risk.schemas import (
    BatchRequest,
    BatchResponse,
    LoanApplication,
    PredictionResponse,
)

HERE = Path(__file__).parent
# change à chaque modification du CSS, pour que le navigateur le recharge
CSS_VERSION = hashlib.md5((HERE / "static" / "style.css").read_bytes()).hexdigest()[:8]
FIELDS = LoanApplication.model_fields
EXAMPLES = {
    "risque": {
        "credit_lines_outstanding": 5,
        "loan_amt_outstanding": 1958.93,
        "total_debt_outstanding": 8228.75,
        "income": 26648.44,
        "years_employed": 2,
        "fico_score": 572,
    },
    "intermediaire": {
        "credit_lines_outstanding": 2,
        "loan_amt_outstanding": 3736.31,
        "total_debt_outstanding": 10222.42,
        "income": 59826.48,
        "years_employed": 3,
        "fico_score": 640,
    },
    "sur": {
        "credit_lines_outstanding": 0,
        "loan_amt_outstanding": 5221.55,
        "total_debt_outstanding": 3915.47,
        "income": 78039.39,
        "years_employed": 5,
        "fico_score": 605,
    },
}


def form_fields() -> list[dict[str, Any]]:
    """Champs du formulaire, à partir du schéma Pydantic."""
    fields = []
    for name, info in FIELDS.items():
        bounds = {type(m).__name__: m for m in info.metadata}
        fields.append(
            {
                "name": name,
                "label": info.title,
                "unit": info.json_schema_extra["unit"],
                "step": "1" if info.annotation is int else "any",
                "min": bounds["Ge"].ge if "Ge" in bounds else bounds["Gt"].gt,
                "max": bounds["Le"].le,
            }
        )
    return fields


FORM_FIELDS = form_fields()
API_DESCRIPTION = """
Envoyez la situation financière d'un client, l'API renvoie son **score de crédit**
(300 à 850), sa **note** (A à D), sa **probabilité de défaut** et l'impact de chaque
variable.

Pour essayer : ouvrez une fonction ci-dessous, puis cliquez sur **Execute**.
Un exemple de client est déjà rempli.

[← Retour à CreditScore](/) · [Présentation de l'API](/developpeurs)
"""
REQUEST_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
CSP = (
    "default-src 'self'; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src https://fonts.gstatic.com; img-src 'self' data:; "
    "form-action 'self'; base-uri 'self'; frame-ancestors "
)


def dashboard(application: LoanApplication, result: dict[str, Any]) -> dict:
    """Données des graphiques affichés avec le résultat."""
    share = (result["credit_score"] - SCORE_MIN) / (SCORE_MAX - SCORE_MIN)
    angle = math.pi * (1 - share)
    max_points = max(abs(c["points"]) for c in result["explanation"]) or 1
    income = application.income or 1
    return {
        "gauge": {
            "share": share * 100,
            "x": 120 + 100 * math.cos(angle),
            "y": 120 - 100 * math.sin(angle),
        },
        "debt_ratio": application.total_debt_outstanding / income * 100,
        "loan_ratio": application.loan_amt_outstanding / income * 100,
        "factors": [
            {**c, "width": abs(c["points"]) / max_points * 100}
            for c in result["explanation"]
        ],
    }


logger = logging.getLogger("credit_risk")
templates = Jinja2Templates(directory=HERE / "templates")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = CreditRiskModel.load(
            settings.model_path, settings.model_card_path, settings.decision_threshold
        )
        logger.info(
            "Modèle chargé : %s v%s (seuil %.3f)",
            app.state.model.card["model_name"],
            app.state.model.version,
            app.state.model.threshold,
        )
        yield

    app = FastAPI(
        title="CreditScore API",
        description=API_DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "Prédiction", "description": "Évaluer un ou plusieurs clients"},
            {"name": "Modèle", "description": "Informations sur le modèle utilisé"},
            {"name": "Supervision", "description": "État du service"},
        ],
        swagger_ui_parameters={
            "defaultModelsExpandDepth": -1,  # cache les schémas techniques
            "docExpansion": "list",
            "tryItOutEnabled": True,
            "displayRequestDuration": True,
        },
    )
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    @app.middleware("http")
    async def observability(request: Request, call_next):
        # on ne reprend l'identifiant du client que s'il est court et sans danger
        request_id = request.headers.get("X-Request-ID", "")
        if not REQUEST_ID.fullmatch(request_id):
            request_id = uuid.uuid4().hex[:12]
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        if settings.frame_ancestors == "'none'":
            response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=()"
        # /docs charge Swagger depuis un CDN : on ne restreint que nos pages
        if not request.url.path.startswith(("/docs", "/redoc")):
            response.headers["Content-Security-Policy"] = CSP + settings.frame_ancestors
        if request.url.path != "/health":
            logger.info(
                "request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
                request_id,
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
            )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "detail": clean_errors(exc.errors()),
                "messages": french_messages(exc.errors()),
            },
        )

    def model(request: Request) -> CreditRiskModel:
        return request.app.state.model

    # API

    @app.get("/health", tags=["Supervision"], summary="Vérifier que le service tourne")
    def health(request: Request):
        return {"status": "ok", "model_version": model(request).version}

    @app.get("/api/v1/model", tags=["Modèle"], summary="Informations sur le modèle")
    def model_info(request: Request):
        """Infos sur le modèle (version, métriques, seuil)."""
        current = model(request)
        return {**current.card, "threshold": current.threshold}

    @app.post(
        "/api/v1/predict",
        response_model=PredictionResponse,
        tags=["Prédiction"],
        summary="Évaluer un client",
    )
    def predict(
        application: LoanApplication,
        request: Request,
        explain: bool = Query(True, description="Inclure l'explication"),
    ):
        """Renvoie le score de crédit, la note, la probabilité de défaut et,
        pour chaque variable, les points gagnés ou perdus sur le score."""
        current = model(request)
        result = current.predict([application], explain=explain)[0]
        return {**result, "model_version": current.version}

    @app.post(
        "/api/v1/predict/batch",
        response_model=BatchResponse,
        tags=["Prédiction"],
        summary="Évaluer plusieurs clients (jusqu'à 1 000)",
    )
    def predict_batch(batch: BatchRequest, request: Request):
        """Même résultat que pour un client (sans l'explication), pour chaque
        client de la liste, dans le même ordre."""
        current = model(request)
        return {
            "model_version": current.version,
            "predictions": current.predict(batch.applications),
        }

    # Interface web

    def render(
        request: Request,
        values: dict,
        result=None,
        errors=None,
        dash=None,
        example=None,
    ):
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "fields": FORM_FIELDS,
                "values": values,
                "result": result,
                "dash": dash,
                "css_version": CSS_VERSION,
                "page": "evaluer",
                "example": example,
                "errors": errors or [],
                "card": model(request).card,
                "threshold": model(request).threshold,
                "version": __version__,
            },
        )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home(request: Request, exemple: str | None = None):
        return render(request, EXAMPLES.get(exemple, {}), example=exemple)

    @app.post("/", response_class=HTMLResponse, include_in_schema=False)
    async def home_submit(request: Request):
        form = await request.form()
        # virgule française acceptée : 1958,93 -> 1958.93
        values = {
            k: str(v).strip().replace(",", ".")
            for k, v in form.items()
            if k in FIELDS and str(v).strip()
        }
        try:
            application = LoanApplication.model_validate(values)
        except ValidationError as exc:
            return render(
                request, values, errors=french_messages(exc.errors()), result=None
            )
        result = model(request).predict([application], explain=True)[0]
        return render(
            request, values, result=result, dash=dashboard(application, result)
        )

    @app.post(
        "/api/v1/fichier",
        tags=["Prédiction"],
        summary="Évaluer un fichier de clients (le plus simple)",
        response_class=FileResponse,
        responses={
            200: {"content": {"text/csv": {}}, "description": "Fichier de résultats"},
            422: {"description": "Fichier illisible ou colonnes manquantes"},
        },
    )
    def predict_file(
        request: Request,
        fichier: UploadFile = File(description="Fichier .csv ou .xlsx"),
    ):
        """Envoyez un fichier CSV ou Excel avec une ligne par client : vous recevez
        le même fichier, complété avec le score, la note et la probabilité de défaut.
        """
        content = fichier.file.read(batch_file.MAX_BYTES + 1)
        try:
            frame = batch_file.read_table(fichier.filename or "", content)
        except batch_file.FileError as exc:
            return JSONResponse(status_code=422, content={"messages": [str(exc)]})
        result = batch_file.score_table(model(request), frame)
        path = batch_file.result_path(batch_file.save(result))
        return FileResponse(
            path, media_type="text/csv", filename="resultats_creditscore.csv"
        )

    # Évaluation d'un fichier de clients

    def render_file_page(request: Request, **context):
        return templates.TemplateResponse(
            request,
            "fichier.html",
            {
                "page": "fichier",
                "card": model(request).card,
                "css_version": CSS_VERSION,
                "fields": FORM_FIELDS,
                **context,
            },
        )

    @app.get("/fichier", response_class=HTMLResponse, include_in_schema=False)
    def file_page(request: Request):
        return render_file_page(request)

    @app.post("/fichier", response_class=HTMLResponse, include_in_schema=False)
    def file_upload(request: Request, fichier: UploadFile = File(...)):
        content = fichier.file.read(batch_file.MAX_BYTES + 1)
        try:
            frame = batch_file.read_table(fichier.filename or "", content)
        except batch_file.FileError as exc:
            return render_file_page(request, error=str(exc))
        result = batch_file.score_table(model(request), frame)
        return render_file_page(
            request,
            filename=fichier.filename,
            summary=batch_file.summarize(result),
            token=batch_file.save(result),
        )

    @app.get("/fichier/modele.csv", include_in_schema=False)
    def file_template():
        return Response(
            batch_file.template_csv(list(EXAMPLES.values())),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": 'attachment; filename="modele_clients.csv"'
            },
        )

    @app.get("/fichier/resultats/{token}", include_in_schema=False)
    def file_results(token: str):
        path = batch_file.result_path(token)
        if path is None:
            raise HTTPException(404, "Résultats introuvables ou expirés.")
        return FileResponse(
            path, media_type="text/csv", filename="resultats_creditscore.csv"
        )

    @app.get("/developpeurs", response_class=HTMLResponse, include_in_schema=False)
    def developers(request: Request):
        application = LoanApplication(**EXAMPLES["risque"])
        response = model(request).predict([application])[0]
        response["model_version"] = model(request).version
        return templates.TemplateResponse(
            request,
            "developpeurs.html",
            {
                "page": "developpeurs",
                "card": model(request).card,
                "css_version": CSS_VERSION,
                "base_url": str(request.base_url).rstrip("/"),
                "request_example": json.dumps(EXAMPLES["risque"], indent=2),
                "response_example": json.dumps(response, indent=2, ensure_ascii=False),
            },
        )

    return app


app = create_app()
