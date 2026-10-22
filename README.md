# Prédiction du défaut de paiement (projet MLOps)

[![CI](https://github.com/Naomi-na2024/prediction-defaut-credit-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/Naomi-na2024/prediction-defaut-credit-mlops/actions/workflows/ci.yml)

Le but du projet est de prédire si un client risque de ne pas rembourser son prêt. J'ai voulu aller au-delà
du notebook : le modèle est servi par une API, l'application tourne dans Docker et les tests sont lancés
automatiquement avec GitHub Actions.

![Aperçu de l'application](docs/apercu-application.png)

## Données

10 000 clients avec 6 variables : nombre de lignes de crédit, montant du prêt restant, dette totale,
revenu, ancienneté professionnelle et score FICO. 18,5 % des clients sont en défaut.

## Modèle

J'ai comparé 4 modèles en validation croisée : régression logistique, arbre de décision, forêt aléatoire
et gradient boosting. La régression logistique donne les meilleurs résultats, et c'est aussi la plus facile
à expliquer, donc c'est celle que j'ai gardée. J'ai aussi testé SMOTE, mais ça n'améliorait pas le modèle.

Résultats sur le jeu de test : AUC de 1,00 et F1 de 99,6 %. Ces scores sont très élevés parce que les
données sont sans doute synthétiques : sur des vraies données, il faudrait s'attendre à moins.

La probabilité de défaut est ensuite convertie en score de crédit de 300 à 850, comme dans une scorecard
bancaire, avec une note de A à D. L'application montre aussi l'impact de chaque variable sur ce score.

Toute l'analyse est dans `notebooks/analyse_et_modelisation.ipynb`.

## Organisation

```
credit_risk/   code de l'application (entraînement, modèle, API, page web)
data/          données
models/        modèle entraîné + ses métriques
notebooks/     analyse et choix du modèle
tests/         tests
docs/          guide d'utilisation
deploy/        mise en ligne sur Hugging Face
```

## Lancer le projet

Avec Docker :

```bash
docker compose up --build
```

Sans Docker :

```bash
pip install -r requirements-dev.txt
uvicorn credit_risk.api:app --port 5000
```

L'application est ensuite sur http://localhost:5000. On peut y évaluer un client, ou importer un fichier
Excel / CSV pour noter tout un portefeuille (50 000 clients en moins de 3 secondes).
La documentation de l'API est sur http://localhost:5000/docs.

Pour réentraîner le modèle et lancer les tests :

```bash
python -m credit_risk.train
pytest
```

Plus de détails dans le [guide d'utilisation](docs/GUIDE_UTILISATION.md).

## Exemple d'appel à l'API

```bash
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"credit_lines_outstanding": 5, "loan_amt_outstanding": 1958.93, "total_debt_outstanding": 8228.75,
       "income": 26648.44, "years_employed": 2, "fico_score": 572}'
```

La réponse donne la probabilité de défaut, le score de crédit, la note et l'impact de chaque variable.

## Outils utilisés

Python, scikit-learn, FastAPI, Docker, GitHub Actions, MLflow, AWS (ECR et ECS).

## Pistes d'amélioration

- Choisir le seuil en fonction du coût réel d'une erreur pour la banque
- Suivre la dérive des données une fois le modèle en production
- Tester le modèle sur des données réelles

## Auteure

Naomi Nandjui - [LinkedIn](https://www.linkedin.com/in/blesson-naomi-nandjui-na)

© 2024-2026 Naomi Nandjui. Tous droits réservés : le code est consultable, mais pas réutilisable
sans mon accord (voir [LICENSE](LICENSE)).
