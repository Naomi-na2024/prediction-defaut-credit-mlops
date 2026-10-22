# Guide d'utilisation

L'application est en ligne : **https://prediction-defaut-credit-mlops.onrender.com**

Rien à installer, elle s'ouvre dans le navigateur, sur ordinateur ou sur téléphone.

## Évaluer un client

Page d'accueil : https://prediction-defaut-credit-mlops.onrender.com

On remplit les 6 champs (ou on clique sur un des exemples), puis « Évaluer ».
La page affiche le score de crédit (300 à 850), la note, la probabilité de défaut, deux ratios
(dette / revenu et prêt / revenu) et l'impact de chaque variable sur le score.

Notes : A de 700 à 850 et B de 620 à 699 (risque faible, en bleu), C de 540 à 619 (risque modéré,
en orange), D de 300 à 539 (risque élevé, en rouge).

Si une valeur sort de ce que le modèle a vu à l'entraînement (par exemple plus de 5 lignes de crédit),
un encadré orange prévient que le résultat est à interpréter avec prudence.

## Évaluer un fichier de clients

Page « Évaluer un fichier » : https://prediction-defaut-credit-mlops.onrender.com/fichier

1. Télécharger le modèle de fichier (`modele_clients.csv`) pour voir les colonnes attendues.
2. Mettre une ligne par client. Les autres colonnes (nom, numéro de client…) sont conservées.
3. Importer le fichier (.csv ou .xlsx, jusqu'à 100 000 clients).

La page affiche un résumé (score moyen, défauts prédits, répartition par note, lignes en erreur)
et un bouton pour télécharger le fichier complété avec le score, la note et la probabilité de défaut.
Le fichier de résultats s'ouvre directement dans Excel et reste disponible une heure.

## Utiliser l'API

- Présentation simple : https://prediction-defaut-credit-mlops.onrender.com/developpeurs
- Pour l'essayer : https://prediction-defaut-credit-mlops.onrender.com/docs, choisir « Évaluer un client »,
  puis « Execute ».

Routes disponibles :
- `POST /api/v1/fichier` : on envoie un fichier CSV ou Excel, on reçoit le fichier de résultats
- `POST /api/v1/predict` : prédiction pour un client
- `POST /api/v1/predict/batch` : prédiction pour plusieurs clients
- `GET /api/v1/model` : infos sur le modèle (version, métriques, seuil)
- `GET /health` : vérifie que l'application tourne

Envoyer un fichier en une ligne de commande :

```bash
curl -F "fichier=@clients.csv" https://prediction-defaut-credit-mlops.onrender.com/api/v1/fichier -o resultats.csv
```

Évaluer un client en Python :

```python
import requests

client = {"credit_lines_outstanding": 5, "loan_amt_outstanding": 1958.93,
          "total_debt_outstanding": 8228.75, "income": 26648.44,
          "years_employed": 2, "fico_score": 572}
url = "https://prediction-defaut-credit-mlops.onrender.com/api/v1/predict"
print(requests.post(url, json=client).json())
```

## Pour les développeurs : faire tourner le projet soi-même

Avec Docker (Docker Desktop doit être ouvert) :

```bash
git clone https://github.com/Naomi-na2024/prediction-defaut-credit-mlops.git
cd prediction-defaut-credit-mlops
docker compose up --build -d
```

L'application s'ouvre alors sur le port 5000 de son ordinateur. Pour l'arrêter : `docker compose down`.

Réentraîner le modèle (compare les modèles, garde le meilleur et met à jour `models/`) :

```bash
pip install -r requirements-dev.txt
python -m credit_risk.train
```

Lancer les tests :

```bash
pytest
ruff check .
```
