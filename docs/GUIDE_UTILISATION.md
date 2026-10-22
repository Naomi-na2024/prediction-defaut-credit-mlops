# Guide d'utilisation

## Lancer l'application

Avec Docker (Docker Desktop doit être ouvert) :

```bash
docker compose up --build -d
```

Pour l'arrêter : `docker compose down`.

Sans Docker :

```bash
python -m venv .venv
.venv\Scripts\activate        # sous Mac/Linux : source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn credit_risk.api:app --port 5000
```

L'application est sur http://localhost:5000.

## Utiliser la page web

On remplit les 6 champs (ou on clique sur un des exemples), puis « Évaluer le dossier ».
La page affiche le score de crédit (300 à 850), la note, la probabilité de défaut, deux ratios
(dette / revenu et prêt / revenu) et l'impact de chaque variable sur le score.

Notes : A de 700 à 850 et B de 620 à 699 (risque faible, en bleu), C de 540 à 619 (risque modéré,
en orange), D de 300 à 539 (risque élevé, en rouge).

Si une valeur sort de ce que le modèle a vu à l'entraînement (par exemple plus de 5 lignes de crédit),
un encadré orange prévient que le résultat est à interpréter avec prudence.

## Évaluer un fichier de clients

Page « Évaluer un fichier » (http://localhost:5000/fichier) :

1. Télécharger le modèle de fichier (`modele_clients.csv`) pour voir les colonnes attendues.
2. Mettre une ligne par client. Les autres colonnes (nom, numéro de client…) sont conservées.
3. Importer le fichier (.csv ou .xlsx, jusqu'à 100 000 clients).

La page affiche un résumé (score moyen, défauts prédits, répartition par note, lignes en erreur)
et un bouton pour télécharger le fichier complété avec le score, la note et la probabilité de défaut.
Le fichier de résultats s'ouvre directement dans Excel et reste disponible une heure.

## Utiliser l'API

La page http://localhost:5000/developpeurs explique simplement l'API, avec un exemple de question et
de réponse. Pour l'essayer, le plus simple est http://localhost:5000/docs : on choisit
« Évaluer un client », puis « Execute ».

Routes disponibles :
- `POST /api/v1/fichier` : on envoie un fichier CSV ou Excel, on reçoit le fichier de résultats
  (`curl -F "fichier=@clients.csv" http://localhost:5000/api/v1/fichier -o resultats.csv`)
- `POST /api/v1/predict` : prédiction pour un client
- `POST /api/v1/predict/batch` : prédiction pour plusieurs clients
- `GET /api/v1/model` : infos sur le modèle (version, métriques, seuil)
- `GET /health` : vérifie que l'application tourne

Exemple en Python :

```python
import requests

client = {"credit_lines_outstanding": 5, "loan_amt_outstanding": 1958.93,
          "total_debt_outstanding": 8228.75, "income": 26648.44,
          "years_employed": 2, "fico_score": 572}
print(requests.post("http://localhost:5000/api/v1/predict", json=client).json())
```

## Réentraîner le modèle

```bash
python -m credit_risk.train
```

Le script compare les modèles, garde le meilleur et met à jour le dossier `models/`.
Avec `--mlflow`, les résultats sont aussi enregistrés dans MLflow (`mlflow ui` pour les voir).

## Tests

```bash
pytest
ruff check .
```

## Problèmes fréquents

- Erreur `dockerDesktopLinuxEngine` : Docker Desktop n'est pas lancé.
- Port 5000 déjà utilisé : changer `"5000:5000"` en `"8080:5000"` dans `docker-compose.yml`.
