# CreditScore : prédiction du défaut de paiement

[![CI](https://github.com/Naomi-na2024/prediction-defaut-credit-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/Naomi-na2024/prediction-defaut-credit-mlops/actions/workflows/ci.yml)

Une application qui estime si un client risque de ne pas rembourser son prêt, et lui donne un score de crédit.

### [Essayer l'application en ligne →](https://prediction-defaut-credit-mlops.onrender.com)

![Aperçu de l'application](docs/apercu-application.png)

## En bref

- 10 000 clients, 6 variables financières
- 4 modèles comparés : la régression logistique est retenue (AUC 1,00, F1 99,6 %)
- Score de 300 à 850, note de A à D, et ce qui a pesé dans la décision
- Un client à la fois, ou tout un fichier Excel / CSV d'un coup
- API, Docker, tests automatiques avec GitHub Actions

## Pour aller plus loin

- L'analyse et le choix du modèle : [notebooks/analyse_et_modelisation.ipynb](notebooks/analyse_et_modelisation.ipynb)
- Le guide d'utilisation (et comment faire tourner le projet soi-même) : [docs/GUIDE_UTILISATION.md](docs/GUIDE_UTILISATION.md)
- La documentation de l'API : https://prediction-defaut-credit-mlops.onrender.com/docs

## Auteure

Naomi Nandjui · [LinkedIn](https://www.linkedin.com/in/blesson-naomi-nandjui-na)

© 2024-2026 Naomi Nandjui. Tous droits réservés (voir [LICENSE](LICENSE)).
