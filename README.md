# Risque systémique et CoVaR — modules Python

> **Données exclusivement synthétiques.** Toutes les institutions, valeurs, rendements, estimations et graphiques sont simulés à des fins pédagogiques. Rien dans ce dépôt ne représente les performances ou le risque réel du secteur bancaire marocain. Aucun service ou code ne télécharge des prix réels.

## Périmètre de cette contribution

Ce dossier contient la contribution Python : génération de données synthétiques, préparation des rendements, GARCH(1,1), DCC-GARCH, VaR, CoVaR/ΔCoVaR, graphiques, exports et tests. Les répertoires et composants hors de ce périmètre restent réservés aux autres membres de l'équipe.

Les cinq établissements simulés sont identifiés par `ATW_SIM`, `BCP_SIM`, `BOA_SIM`, `CIH_SIM` et `CDM_SIM`. `MASI_SIM` est un indice de système simulé. Pour l'analyse CoVaR d'une institution, le proxy pair système exclut l'institution évaluée afin d'éviter une relation partie-tout directe.

## Architecture

- `src/synthetic_data.py` : générateur déterministe de cinq années environ de prix, rendements et indice système simulés.
- `src/preprocessing.py`, `src/data.py` : nettoyage, rendements logarithmiques, alignement, contrôles de provenance CSV.
- `src/garch.py` : estimation GARCH(1,1), volatilité, résidus standardisés et diagnostics.
- `src/dcc_garch.py` : estimation DCC sur résidus standardisés, contraintes paramétriques et matrices dynamiques.
- `src/var.py`, `src/covar.py` : VaR historique/gaussienne, régression quantile CoVaR, ΔCoVaR et fenêtres roulantes.
- `src/systemic_risk.py` : moyenne des corrélations dynamiques et classement descriptif CoVaR.
- `src/visualization.py`, `src/export.py` : figures et sorties CSV/JSON marquées synthétiques.
- `tests/` : tests unitaires dédiés aux modules Python.
- `notebooks/01` à `04` : démonstrations pédagogiques réutilisant les modules.

## Exécution avec Docker

Docker Desktop avec le moteur Linux et Docker Compose doit être démarré. Python n'a pas besoin d'être installé sur Windows.

Dans PowerShell, depuis le dossier du projet :

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

Définissez `POSTGRES_PASSWORD` et `JUPYTER_TOKEN` avec des valeurs privées non vides. `.env` est ignoré par Git. La configuration collective démarre Python/Jupyter, PostgreSQL et Odoo :

```powershell
docker compose up --build -d
docker compose ps
```

- JupyterLab : <http://localhost:8888>, authentification avec le jeton de `.env`.
- Odoo : <http://localhost:8069>, service d'infrastructure pour la contribution d'un autre membre; aucun code Odoo n'est inclus ici.
- PostgreSQL : port local `5432`, avec stockage dans un volume Docker persistant. Odoo le contacte via le nom d'hôte `db`.
- Service Python : les dépendances sont installées dans l'image Docker, le jeu canonique est généré s'il manque et les tests s'exécutent dans ce conteneur.
- Les fichiers du projet sont montés sous `/workspace`; les CSV sont persistés sous `data/synthetic/`.

Commandes utiles :

```powershell
docker compose exec python pytest -q
docker compose exec python pytest -q tests/test_dcc_garch.py
docker compose exec python python -m src.synthetic_data --if-missing
docker compose exec python python -m src.synthetic_data --force
docker compose down
```

`--force` régénère explicitement les quatre CSV avec la graine par défaut. Sans cette option, les quatre fichiers existants sont conservés; si le jeu est partiel, le chargeur refuse la régénération et préserve les fichiers présents. La création initiale n'a lieu que lorsque les quatre fichiers sont absents.

## Notebooks

Ouvrir JupyterLab et exécuter dans l'ordre :

1. `notebooks/01_data_preparation.ipynb`
2. `notebooks/02_garch_model.ipynb`
3. `notebooks/03_dcc_garch.ipynb`
4. `notebooks/04_var_covar.ipynb`

Chaque notebook utilise les mêmes CSV canoniques et les modules `src/`; les calculs ne sont pas recopiés dans les cellules.

## Données synthétiques et limites

La variante synthétique historique est conservée dans `data/synthetic/`. Le pipeline actif utilise les quatre fichiers réels canonique dans `data/real/`: `bank_prices.csv`, `bank_returns.csv`, `market_index.csv` et `financial_dataset.csv`. Les données réelles proviennent des CSV locaux importés dans `old-erp/data/raw/`. Les données synthétiques restent disponibles via le générateur NumPy `default_rng(20251115)`, des chocs corrélés, une dynamique de variance GARCH-like et deux fenêtres de stress simulées. Le premier rendement correspond au passage du niveau de base implicite (100 pour une banque, 1000 pour l'indice) au premier prix daté; le calcul à partir des prix observés commence donc à la deuxième date. Les prix sont strictement positifs. Le calendrier de jours ouvrés est générique et ne représente pas les jours fériés ou les séances de la Bourse de Casablanca.

Les exports ajoutent `data_status` avec la mention `SYNTHETIC / SIMULATED DATA — NOT REAL MARKET DATA`. L'import des rendements CSV impose aussi ce marqueur. Il ne faut pas interpréter les paramètres, estimations, classements ni les épisodes de stress comme des constats sur une banque réelle.

## Méthodes et conventions

- Rendement : `r_t = ln(P_t / P_(t-1))`. En cas de prix manquant, le prétraitement conserve l'écart et ne calcule pas un rendement multi-jour comme s'il était journalier.
- GARCH : `h_t = omega + alpha * epsilon_(t-1)^2 + beta * h_(t-1)`. Le modèle ajusté est à moyenne conditionnelle nulle avec innovations normales; la persistance `alpha + beta` et des diagnostics de résidus sont rapportés.
- DCC : les résidus standardisés produits par GARCH alimentent `Q_t = (1-a-b)Q_bar + a z_(t-1)z'_(t-1) + bQ_(t-1)`. La normalisation donne `R_t`. L'optimisation impose `a >= 0`, `b >= 0`, `a+b < 1`. Une corrélation n'est pas une causalité.
- VaR : `-q_alpha(r)` pour la méthode historique et `-(mu + Phi^-1(alpha)*sigma)` sous hypothèse gaussienne; une valeur négative peut indiquer que le seuil de quantile ne correspond pas à une perte.
- CoVaR : régression quantile du rendement moyen des pairs (hors institution cible) sur le rendement de l'institution. L'état de détresse évalue le quantile inférieur bancaire; l'état normal utilise sa médiane. `ΔCoVaR = CoVaR_détresse - CoVaR_médiane`, exprimé en rendement, généralement plus négatif en détresse.
- La version glissante utilise seulement les observations de chaque fenêtre terminée à la date affichée. Les modèles statiques décrivent l'échantillon utilisé et ne sont pas des prévisions hors échantillon.

Les résultats sont des démonstrations statistiques reproductibles, pas des mesures de risque opérationnelles ni des conseils financiers.

## Développement

Les versions Python et dépendances sont fixées dans `docker/python/Dockerfile` et `requirements.txt`. Exécuter les tests Python dans Docker avec `docker compose exec python pytest -q`. La configuration Compose complète reste partagée entre les membres du projet; Odoo et PostgreSQL y sont présents comme services, sans implémentation des fonctionnalités d'un autre membre. Le rapport de corrections et de validation Python est dans [docs/python_covar_validation_report.md](docs/python_covar_validation_report.md).
