FROM python:3.11-slim

WORKDIR /app

# Installer d'abord les dépendances (mieux mis en cache par Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier uniquement ce dont l'application a besoin
COPY app.py .
COPY models/ models/
COPY templates/ templates/
COPY static/ static/

EXPOSE 5000

# Gunicorn : serveur de production (plus robuste que le serveur Flask de développement)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
