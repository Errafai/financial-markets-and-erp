# Rapport de correction et de validation Python CoVaR

**Date :** 15 novembre 2026
**Périmètre :** contribution Python quantitative uniquement. Aucune modification du fichier `docker-compose.yml`, du code Odoo ou des répertoires réservés aux autres membres.

> Les jeux de données et observations vérifiés sont synthétiques et destinés à la démonstration académique. Ils ne représentent pas des données ou mesures de risque réelles du secteur bancaire marocain.

> **Mise à jour de validation :** voir la dernière section « Validation complémentaire » pour les exécutions réalisées après la première rédaction. Ses résultats remplacent les statuts NOT RUN antérieurs lorsqu’ils portent sur la même vérification.

## A. Corrections réalisées

### 1. VaR historique et convention de quantile

- **Fichiers :** `src/var.py`, `tests/test_var.py`, `docs/python_guide.md`, `docs/methodology.md`.
- La formule existante, `-np.quantile(returns, alpha)`, est conservée. NumPy utilise par défaut le quantile linéaire interpolé.
- Pour l’échantillon du test (5 rendements à -10 %, 15 à 0 % et 30 à +10 %), le rang à 10 % se trouve entre -10 % et 0 %. Le quantile interpolé est -1 %, donc la VaR est **+1 %**, et non +10 %.
- L’attente du test a été corrigée à 0,01. Deux tests complémentaires couvrent un quantile de perte, qui donne une VaR positive, et un quantile de gain, qui donne une VaR négative selon la formule. La VaR n’est pas tronquée artificiellement à zéro.

### 2. Alignement des résidus GARCH avant DCC

- **Fichiers :** `src/garch.py`, `src/dcc_garch.py`, `notebooks/03_dcc_garch.ipynb`, `tests/test_garch.py`, `tests/test_dcc_garch.py`, `docs/python_guide.md`, `docs/methodology.md`.
- Ajout de `align_standardized_residuals()`: la fonction conserve uniquement l’intersection des dates des séries, convertit les valeurs infinies en valeurs manquantes puis écarte les lignes non complètes. Elle n’impute pas de zéro, n’interpole pas et préserve les dates.
- La fonction rejette les résultats GARCH non convergents ou non admissibles et exige au moins 30 dates communes finies avant DCC.
- Le notebook 03 utilise désormais cette fonction et vérifie que les dates renvoyées par DCC correspondent à l’index aligné.
- Les tests ajoutés couvrent des calendriers différents et un résidu infini.

### 3. Protection des fichiers CSV canoniques

- **Fichiers :** `src/data.py`, `src/synthetic_data.py`, `tests/test_data.py`, `README.md`, `docs/python_guide.md`, `docs/methodology.md`.
- Un jeu absent (aucun CSV présent) peut être créé lors du premier chargement. Un jeu complet est chargé sans être réécrit.
- Un jeu partiel entraîne maintenant une erreur explicite; les fichiers présents restent intacts. La régénération volontaire des quatre fichiers nécessite `force=True` ou l’option CLI `--force`.
- Le comportement CLI par défaut ne force plus l’écrasement. Les tests ajoutés couvrent la création initiale, le chargement complet, le refus d’un jeu partiel, la conservation des fichiers et la régénération explicitement forcée.

### 4. Admissibilité et convergence GARCH

- **Fichiers :** `src/garch.py`, `tests/test_garch.py`, `docs/python_guide.md`, `docs/methodology.md`.
- Un ajustement admissible doit converger, avoir `omega > 0`, `alpha >= 0`, `beta >= 0` et `alpha + beta < 1`.
- Les échecs de convergence ou d’admissibilité lèvent `GARCHFitError`, qui expose les paramètres estimés et les diagnostics disponibles. Aucun paramètre de substitution n’est fabriqué.
- Les tests couvrent un ajustement non convergent, un ajustement non stationnaire et les paramètres de variance non positifs/négatifs.

### 5. Validation des paramètres DCC

- **Fichiers :** `src/dcc_garch.py`, `tests/test_dcc_garch.py`.
- Le contrôle DCC rejette maintenant aussi les paramètres `NaN` ou infinis, en plus des valeurs qui violent `a >= 0`, `b >= 0`, `a + b < 1`.
- Les cas non finis ont été ajoutés aux tests paramétrés.

## B. Tests automatisés et validations exécutées

La suite contient **43 fonctions `test_*`, représentant 49 cas** après expansion des deux tests paramétrés. Aucun résultat de suite pytest n’est déclaré PASS : le moteur Docker ne permettait pas de lancer le conteneur. Le lanceur de tests de l’éditeur n’a pas découvert de tests.

| Fichier / test suite | Fonctions | Cas pytest | Statut | Commentaire |
|---|---:|---:|---|---|
| `tests/test_data.py` | 12 | 12 | NOT RUN | Docker indisponible; cas de protection CSV exercés séparément par smoke check. |
| `tests/test_garch.py` | 8 | 10 | NOT RUN | Docker indisponible; alignement et rejet non stationnaire exercés séparément avec dépendances externes simulées. |
| `tests/test_dcc_garch.py` | 6 | 10 | NOT RUN | Docker indisponible; filtre DCC testé séparément sur résidus synthétiques corrélés. |
| `tests/test_var.py` | 6 | 6 | NOT RUN | Docker indisponible; les exemples de quantiles ont été vérifiés directement. |
| `tests/test_covar.py` | 6 | 6 | NOT RUN | `statsmodels` absent de l’interpréteur local; estimation quantile non exécutée. |
| `tests/test_systemic_risk.py` | 3 | 3 | NOT RUN | Docker indisponible. |
| `tests/test_export.py` | 1 | 1 | NOT RUN | Docker indisponible. |
| `tests/test_visualization.py` | 1 | 1 | NOT RUN | Docker indisponible; `seaborn` absent de l’interpréteur local. |
| **Total** | **43** | **49** | **NOT RUN** | Pas de verdict pytest global. |

Smoke checks exécutés hors pytest :

| Vérification | Statut | Résultat observé |
|---|---|---|
| Générateur et prétraitement | PASS | 1 304 dates ouvrées, 5 banques simulées et `MASI_SIM`; répétabilité confirmée, prix finis/positifs, volatilité de stress supérieure; rendements calculés à partir des prix identiques aux rendements générés à partir de la seconde date. |
| Chargement et sûreté CSV | PASS | Jeu absent généré en quatre fichiers dans un répertoire temporaire du projet; jeu complet préservé; jeu partiel refusé sans altérer le fichier existant; `force=True` régénère explicitement les quatre fichiers. |
| VaR historique | PASS | L’échantillon litigieux renvoie 0,01 selon l’interpolation linéaire; un quantile de perte donne une valeur positive et un échantillon de gains garde une valeur négative selon la formule. |
| DCC sur résidus synthétiques corrélés | PASS | Optimiseur convergé; contraintes respectées; 250 matrices finies, dates conservées et valeur propre minimale observée d’environ 0,232. Les paramètres non finis et invalides ont été rejetés. Ce smoke check ne remplace pas un ajustement GARCH réel en amont. |
| Alignement et rejet GARCH | PASS* | 37 dates communes finies conservées après intersection/calendriers manquants et suppression d’une valeur infinie; un résultat non stationnaire a été rejeté en conservant paramètres et diagnostics. |
| Syntaxe des fichiers Python modifiés | PASS | Pylance n’a signalé aucune erreur de syntaxe dans les sources et tests modifiés. |
| Structure JSON des notebooks | PASS | Les quatre fichiers sont des notebooks nbformat valides, avec deux cellules de code et sans sorties enregistrées. Cela ne prouve pas leur exécution. |

\*Smoke check ciblé de logique : les imports `arch` et `statsmodels` ont été remplacés par des modules temporaires factices pour atteindre les fonctions d’alignement et d’admissibilité. L’estimation GARCH n’a pas été effectuée dans ce check.

## C. Notebooks

| Notebook | Structure | Exécution |
|---|---|---|
| `notebooks/01_data_preparation.ipynb` | JSON valide; 2 cellules de code; aucune sortie enregistrée. | NOT RUN |
| `notebooks/02_garch_model.ipynb` | JSON valide; 2 cellules de code; aucune sortie enregistrée. | NOT RUN |
| `notebooks/03_dcc_garch.ipynb` | JSON valide; alignement explicite des résidus; 2 cellules de code; aucune sortie enregistrée. | NOT RUN |
| `notebooks/04_var_covar.ipynb` | JSON valide; 2 cellules de code; aucune sortie enregistrée. | NOT RUN |

Le notebook 03 requiert notamment `arch`, `statsmodels` et `seaborn` dans l’environnement Python du projet. Il n’a pas été exécuté de bout en bout.

## D. État de la chaîne quantitative

| Étape | Validation réelle |
|---|---|
| Génération synthétique reproductible | Exécutée en mémoire et vérifiée. |
| Chargement des CSV et protection contre un jeu partiel | Exécutés sur répertoires temporaires; aucun CSV canonique n’a été généré ou remplacé dans `data/synthetic/`. |
| Calcul de rendements depuis les prix | Exécuté et comparé aux rendements générés; concordance à partir de la deuxième date. |
| Ajustement GARCH sur chaque banque | Non exécuté : `arch` absent localement et Docker indisponible. |
| Extraction et alignement de résidus GARCH réels | La logique d’alignement a été exercée sur séries de test, mais aucun résidu provenant d’un vrai ajustement GARCH n’a été produit. |
| Estimation DCC et matrices conditionnelles | Filtre/optimiseur exercés sur résidus synthétiques corrélés, sans GARCH préalable. Le pipeline complet GARCH→DCC reste à exécuter. |
| VaR | Formule et quantiles ciblés exercés directement; tests pytest non exécutés. |
| CoVaR, ΔCoVaR et fenêtres glissantes | Non exécutés : `statsmodels` absent localement. |
| Visualisation et export | Modules/tests non exécutés dans cette validation. |

## E. Docker

- `docker compose config --quiet` : **PASS**; le YAML est accepté par Compose.
- `docker info` et les deux tentatives de test (`docker compose run --rm python pytest -v` puis `pytest --cov=src --cov-report=term-missing`) : **échec avant le lancement du conteneur**. Docker Desktop a renvoyé une erreur **HTTP 500 Internal Server Error** sur son API Linux (`dockerDesktopLinuxEngine`, notamment la route `/networks`).
- Aucun service, environnement Python conteneurisé ou accès Jupyter n’a été démarré/vérifié.
- `docker-compose.yml` a été laissé inchangé; aucun service n’a été supprimé.

## F. Fichiers modifiés, créés ou supprimés pendant cette correction

**Modifiés :**

- `README.md`
- `docs/methodology.md`
- `docs/python_guide.md`
- `notebooks/03_dcc_garch.ipynb`
- `src/data.py`
- `src/dcc_garch.py`
- `src/garch.py`
- `src/synthetic_data.py`
- `src/var.py`
- `tests/test_data.py`
- `tests/test_dcc_garch.py`
- `tests/test_garch.py`
- `tests/test_var.py`

**Créé :**

- `docs/python_covar_validation_report.md`

**Supprimé :** aucun fichier.
**Non modifié :** `docker-compose.yml`, l’addon Odoo et les dossiers réservés aux autres membres.

À noter : au début de ce travail, les fichiers du projet apparaissaient comme non suivis dans Git. La liste ci-dessus décrit les fichiers édités lors de cette correction, et non un diff Git entre commits.

## G. Limites et vérifications restantes

- Les tests pytest n’ont pas été exécutés; il faut relancer les deux commandes dans Docker après rétablissement du moteur.
- L’interpréteur sélectionné est Python 3.13 et ne résout pas localement `arch`, `statsmodels`, `seaborn` et `pytest`. Les dépendances figurent dans `requirements.txt`; elles n’ont pas été installées sur Windows.
- Les notebooks n’ont pas été exécutés; leur validité JSON ne confirme ni leur exécution ni leurs sorties.
- Le contrôle GARCH testé par mock ne certifie pas la convergence statistique de l’estimateur `arch` sur le jeu synthétique.
- CoVaR, ΔCoVaR, résultats glissants, visualisations et exports restent à valider dans l’environnement Docker.
- L’addon Odoo et l’EWS ne sont pas dans le périmètre et ne sont pas livrés dans cette contribution.

## H. Conclusion

Les quatre corrections demandées ont été appliquées sans toucher à l’architecture Docker collective ni aux parties des autres membres. Des smoke checks réels confirment la reproductibilité du générateur, la politique de protection des CSV, le résultat de la VaR avec interpolation linéaire et le comportement de la logique DCC/alignment ciblée.

La contribution ne peut pas encore être déclarée entièrement validée de bout en bout : l’estimation GARCH, CoVaR et les tests automatisés complets restent à exécuter. Le blocage immédiat est le HTTP 500 du moteur Docker; les dépendances manquantes dans l’interpréteur local empêchent de contourner ce blocage sans changer l’environnement demandé.

## Validation complémentaire — 15 novembre 2026

Cette section remplace les statuts précédents pour les validations effectuées ci-dessous. Pendant cette étape, aucun module, test, notebook, fichier de dépendances, Compose ou fichier de données n’a été modifié. Seul ce rapport est mis à jour.

### Environnement et commandes

Docker reste indisponible : le contexte actif est `desktop-linux`; `docker info --format '{{.ServerVersion}}'` a renvoyé HTTP 500 de l’API Engine sur `dockerDesktopLinuxEngine/v1.54/info`. `docker compose config --quiet` a réussi. Le démarrage des services n’a pas été tenté après confirmation de l’indisponibilité du moteur.

Un venv isolé a été créé hors du projet, dans le dossier de session :

```powershell
C:\Users\safaa\AppData\Local\Microsoft\WindowsApps\python3.11.exe -m venv C:\Users\safaa\.copilot\session-state\485ed53d-7f51-413b-921a-ebaeebec9b88\covar-validation-venv
C:\Users\safaa\.copilot\session-state\485ed53d-7f51-413b-921a-ebaeebec9b88\covar-validation-venv\Scripts\python.exe -m pip install -r C:\Users\safaa\OneDrive\Desktop\Projet_ERP\requirements.txt
C:\Users\safaa\.copilot\session-state\485ed53d-7f51-413b-921a-ebaeebec9b88\covar-validation-venv\Scripts\python.exe -m pip check
```

Installation : **PASS**. `pip check` : **No broken requirements found**. Versions vérifiées : Python 3.11.9, arch 7.2.0, matplotlib 3.9.4, NumPy 2.1.3, pandas 2.2.3, SciPy 1.15.2, seaborn 0.13.2, statsmodels 0.14.4, pytest 8.3.4, JupyterLab 4.3.5.

Commande exécutée depuis la racine du projet :

```powershell
C:\Users\safaa\.copilot\session-state\485ed53d-7f51-413b-921a-ebaeebec9b88\covar-validation-venv\Scripts\python.exe -m pytest -v tests
```

Résultat exact : **49 cas collectés; 45 passed, 4 failed, 14 warnings, en 69.47 s**. Aucun test n’a été sauté ou réécrit durant la validation. Le calcul de couverture n’a pas été exécuté : `pytest-cov` n’est pas déclaré dans le fichier de dépendances du projet.

### Résultats individuels des 49 cas

| Test | Résultat | Détail en cas d’échec |
|---|---|---|
| `test_quantile_regression_covar_and_delta_are_consistent` | PASS | |
| `test_rolling_covar_uses_trailing_window_dates` | FAIL | Statsmodels a atteint 10 000 itérations et émis `IterationLimitWarning`; la fonction remonte `RuntimeError: Quantile regression failed to converge.` |
| `test_covar_rejects_invalid_quantile_and_short_sample` | PASS | |
| `test_leave_one_out_proxy_excludes_target_bank` | PASS | |
| `test_covar_rejects_duplicate_indices_and_constant_series` | PASS | |
| `test_rolling_covar_first_output_does_not_depend_on_future_observations` | PASS | |
| `test_synthetic_generation_is_reproducible_and_positive` | PASS | |
| `test_data_loader_writes_only_missing_canonical_files` | PASS | |
| `test_data_loader_generates_an_initial_dataset_when_all_files_are_absent` | PASS | Écrit uniquement dans le répertoire temporaire pytest. |
| `test_data_loader_refuses_partial_dataset_without_overwriting_existing_files` | PASS | |
| `test_save_refuses_partial_dataset_unless_force_is_explicit` | PASS | |
| `test_force_explicitly_regenerates_partial_dataset` | PASS | Les CSV sont dans le répertoire temporaire pytest, pas dans `data/synthetic/`. |
| `test_loader_rejects_csv_missing_synthetic_provenance` | PASS | |
| `test_price_cleaning_and_log_returns_handle_duplicates_and_missing` | PASS | |
| `test_nonpositive_prices_are_rejected` | PASS | |
| `test_returns_do_not_bridge_a_missing_price_observation` | FAIL | Attendu `NaN` au 2024-01-03; observé `0.1906203596086497` (log(121/100)). Le prétraitement retire la ligne entièrement manquante avant le décalage, créant un rendement qui franchit le manque. |
| `test_csv_import_rejects_duplicate_dates` | PASS | |
| `test_csv_import_requires_an_explicit_synthetic_data_label` | PASS | |
| `test_dcc_filter_obeys_correlation_matrix_invariants` | PASS | |
| `test_dcc_rejects_invalid_parameters[-0.1-0.5]` | PASS | |
| `test_dcc_rejects_invalid_parameters[0.2--0.1]` | PASS | |
| `test_dcc_rejects_invalid_parameters[0.5-0.5]` | PASS | |
| `test_dcc_rejects_invalid_parameters[nan-0.5]` | PASS | |
| `test_dcc_rejects_invalid_parameters[0.2-inf]` | PASS | |
| `test_dcc_estimation_respects_parameter_constraints` | PASS | |
| `test_upper_triangle_does_not_duplicate_pairs` | PASS | |
| `test_dcc_uses_supplied_standardized_residuals_without_restandardizing` | PASS | |
| `test_dcc_fit_rejects_duplicate_dates_and_constant_series` | PASS | |
| `test_export_writes_csv_and_iso_json_with_synthetic_label` | PASS | Fichiers dans le répertoire temporaire pytest. |
| `test_garch_returns_valid_volatility_and_standardized_residuals` | PASS | GARCH 1,1 exécuté sur données générées en mémoire. |
| `test_garch_rejects_short_or_constant_input` | PASS | |
| `test_garch_rejects_orders_outside_project_model` | FAIL | La fonction lève le `ValueError` attendu; le motif regex `GARCH(1,1) only` interprète les parenthèses comme un groupe et ne correspond pas au message littéral. |
| `test_garch_panel_residuals_align_on_common_dates_without_filling` | PASS | |
| `test_dcc_alignment_rejects_inadmissible_garch_result` | PASS | |
| `test_garch_rejects_nonstationary_fit_and_preserves_estimates` | PASS | |
| `test_garch_rejects_nonconvergent_fit_and_preserves_estimates` | PASS | |
| `test_garch_rejects_nonpositive_variance_parameters[parameters0]` | PASS | |
| `test_garch_rejects_nonpositive_variance_parameters[parameters1]` | PASS | |
| `test_garch_rejects_nonpositive_variance_parameters[parameters2]` | PASS | |
| `test_risk_ranking_places_largest_downside_first` | PASS | |
| `test_correlation_summary_averages_unique_off_diagonal_pairs` | PASS | |
| `test_correlation_summary_rejects_mismatched_metadata` | PASS | |
| `test_var_estimators_return_positive_loss_values` | PASS | |
| `test_var_validates_tail_probability_and_observations` | PASS | |
| `test_parametric_var_rejects_zero_volatility` | FAIL | Avec 100 valeurs `0.01`, l’écart-type pandas observé est `1.7434626923745965e-18`, donc strictement positif; `parametric_var` renvoie `-0.009999999999999995` au lieu de rejeter l’entrée quasi constante. |
| `test_var_known_small_sample_quantile` | PASS | |
| `test_historical_var_is_positive_when_lower_tail_is_a_loss` | PASS | |
| `test_historical_var_keeps_sign_if_selected_quantile_is_a_gain` | PASS | |
| `test_plot_labels_synthetic_data_and_saves_image` | PASS | Image dans le répertoire temporaire pytest. |

### Notebooks

| Notebook CoVaR | Statut | Motif |
|---|---|---|
| `notebooks/01_data_preparation.ipynb` | NOT EXECUTED | Aucun CSV canonique existant dans `data/synthetic/`; le chargement par défaut en créerait. |
| `notebooks/02_garch_model.ipynb` | NOT EXECUTED | Même motif; ne pas générer les CSV contre la consigne. |
| `notebooks/03_dcc_garch.ipynb` | NOT EXECUTED | Même motif. |
| `notebooks/04_var_covar.ipynb` | NOT EXECUTED | Même motif. |

Les quatre notebooks sont restés inchangés. Une exécution en mémoire avec des données nouvellement générées n’a pas été substituée à une exécution avec les données existantes demandées.

### État des corrections précédentes

- **VaR historique :** le cas litigieux passe avec la quantile linéaire et la valeur attendue 0,01; les tests de signe passent. Le test de volatilité constante met toutefois en évidence une tolérance manquante pour un écart-type de l’ordre de `1.7e-18`. Aucun code n’a été modifié pour corriger cela, conformément à la consigne.
- **Alignement DCC :** tests d’intersection de dates et de paramètres invalide/non fini passés. La garantie de préservation des dates testée avec des résidus GARCH représentatifs est validée.
- **Protection CSV :** création initiale, dataset complet, refus du dataset partiel et `force=True` explicite passent tous en répertoires temporaires. Aucun CSV canonique du projet n’a été créé ni régénéré.
- **Admissibilité GARCH :** convergence, non-stationnarité, paramètres de variance invalides et rejet avant DCC passent; estimation GARCH 1,1 sur les données simulées en mémoire passe.

### Fichiers et vérification de périmètre

- **Fichier durable du projet modifié durant cette validation :** uniquement `docs/python_covar_validation_report.md`.
- L’environnement virtuel et ses dépendances sont sous le dossier de session, hors du projet. Les tests ont utilisé leurs répertoires temporaires pour les fixtures CSV, images et exports.
- Pytest a généré des artefacts de cache ignorés par Git : `.pytest_cache/`, `src/__pycache__/` et `tests/__pycache__/`. Aucun fichier source ni configuration n’a été modifié par l’exécution des tests.
- SHA-256 de `docker-compose.yml` avant et après validation : `D9A2F74C3CDE8BF0305382D25AB44878E95D930AC9AA59FD287DE3090DA4AB35` (identique). `docker compose config --quiet` reste valide.
- `odoo_addons` contient toujours **0 fichier**.
- `data/synthetic/` contient toujours uniquement `.gitkeep` (SHA-256 inchangé : `9588228E24340BCACB15CCA4FFC50EB0598791EBF694A38A347E63AFBB65130F`); **0 CSV canonique** n’a été touché, supprimé ou régénéré.
- Les fichiers du projet apparaissaient tous non suivis dans Git avant cette validation. Le statut Git ne fournit donc pas de baseline commit/diff; aucune opération Git de modification n’a été effectuée.

### Conclusion actualisée

La suite complète est désormais réellement exécutée dans un venv Python 3.11, mais **elle n’est pas entièrement verte : 45/49 PASS et 4 FAIL**. Deux échecs signalent des défauts de comportement (rendement calculé à travers un prix manquant; VaR paramétrique qui accepte un écart-type quasi nul), un concerne la convergence quantile sur une petite fenêtre, et un est un motif regex de test qui ne correspond pas au message littéral. Ils restent ouverts car cette étape interdisait les modifications de code et de tests.

Les notebooks sont **NOT EXECUTED** : aucun jeu canonique existant n’est disponible, et leur exécution standard générerait des CSV dans le projet, ce qui était interdit. Compose et l’addon Odoo sont préservés. Docker n’a pas démarré en raison de son erreur HTTP 500.

## Validation finale — corrections, données et notebooks

Cette section remplace les résultats antérieurs pour les éléments revérifiés dans la deuxième phase. Toutes les valeurs de marché ci-dessous proviennent exclusivement du générateur synthétique du projet; elles ne sont pas des données bancaires réelles.

### Corrections des quatre échecs

1. **Régression quantile glissante.** Statsmodels IRLS reste le solveur principal. Si Statsmodels lève `IterationLimitWarning` ou `ConvergenceWarning`, `src/covar.py` tente explicitement une programmation linéaire SciPy HiGHS, formulée pour minimiser la même perte quantile pinball. `CoVaRResult.solver` indique le solveur utilisé. Si les deux solveurs échouent, une erreur explicite est levée; aucun résultat non convergent n’est accepté silencieusement. Un test force le signal de non-convergence IRLS et vérifie que le résultat provient de HiGHS.
2. **Prix manquant.** `clean_price_data()` conserve maintenant les dates entièrement manquantes jusqu’au calcul du rendement. Après un prix manquant, le rendement au retour d’une cotation demeure manquant au lieu de calculer une variation sur plusieurs séances. Le test de non-franchissement du manque passe; l’alignement ultérieur continue à écarter les rendements manquants selon la politique existante.
3. **Regex GARCH.** Seul le motif du test est corrigé : le texte littéral `GARCH(1,1) only` est recherché via `re.escape`. Le message et le comportement de validation du modèle restent inchangés.
4. **Volatilité quasi nulle.** La VaR paramétrique rejette les observations exactement constantes en vérifiant leur étendue (`np.ptp`), plutôt que de traiter un minuscule écart-type d’arrondi comme une vraie variance. Une série avec de très petites variations effectivement observées reste acceptée. Une valeur d’écart-type positive fournie explicitement reste admise même si les données d’exemple sont constantes. Trois cas de test couvrent ces comportements.

### Données synthétiques canoniques

Avant la génération, `data/synthetic/` ne contenait que `.gitkeep` et aucun CSV. Les quatre CSV canoniques ont été générés une seule fois, à la demande explicite, sans option d’écrasement, avec la graine `20251115` définie dans `src/config.py`. La commande exécutée après vérification de l’absence de tout CSV était :

```powershell
C:\Users\safaa\.copilot\session-state\485ed53d-7f51-413b-921a-ebaeebec9b88\covar-validation-venv\Scripts\python.exe -m src.synthetic_data --seed 20251115 --if-missing
```

Il s’agit de 1 304 dates ouvrées du 2021-01-01 au 2025-12-31, pour `ATW_SIM`, `BCP_SIM`, `BOA_SIM`, `CIH_SIM`, `CDM_SIM` et l’indice proxy `MASI_SIM`. Chaque CSV porte la colonne `data_status` avec `SYNTHETIC / SIMULATED DATA — NOT REAL MARKET DATA`.

Les quatre fichiers ont des dates uniques et ordonnées, le même calendrier, les colonnes attendues et la provenance synthétique vérifiée sur chaque ligne. Les rendements recalculés depuis les prix coïncident avec les rendements générés à partir de la deuxième date à `rtol=1e-10`, `atol=1e-12`; l’écart absolu maximal observé est `5.95e-16`, dû aux arrondis flottants. Les fichiers canoniques n’ont pas été régénérés pendant les exécutions des notebooks.

### Tests, dépendances et avertissements

L’environnement reste le venv Python 3.11.9 hors projet. `requirements.txt` a été installé sans modification; `pip check` indique `No broken requirements found`. `pytest-cov 7.1.0` et `coverage 7.17.3` ont été ajoutés uniquement à ce venv avec `C:\Users\safaa\.copilot\session-state\485ed53d-7f51-413b-921a-ebaeebec9b88\covar-validation-venv\Scripts\python.exe -m pip install pytest-cov`; le fichier de dépendances du projet n’a pas été modifié.

Commandes pytest exécutées à la racine du projet :

```powershell
C:\Users\safaa\.copilot\session-state\485ed53d-7f51-413b-921a-ebaeebec9b88\covar-validation-venv\Scripts\python.exe -m pytest -v
C:\Users\safaa\.copilot\session-state\485ed53d-7f51-413b-921a-ebaeebec9b88\covar-validation-venv\Scripts\python.exe -m pytest --cov=src --cov-report=term-missing
```

Les **49 cas présents au début de cette phase** passent tous. Trois cas supplémentaires ont été ajoutés pour rendre vérifiables le repli HiGHS et les deux comportements de volatilité; la suite compte donc maintenant **52 cas : 52 passed, 0 failed, 0 skipped**. Les deux commandes ci-dessus ont chacune exécuté les 52 cas. Le rapport de couverture indique **84 %** au total (472/562 lignes exécutées); par module : `covar.py` 90 %, `data.py` 74 %, `dcc_garch.py` 82 %, `export.py` 95 %, `garch.py` 86 %, `preprocessing.py` 81 %, `synthetic_data.py` 86 %, `systemic_risk.py` 77 %, `var.py` 96 %, `visualization.py` 70 % et `config.py` 100 %.

Les **14 warnings pytest** persistent dans les deux commandes et ont tous la même origine, hors logique métier : `PyparsingDeprecationWarning` venant de l’initialisation de Matplotlib 3.9.4. Détail : 1 appel déprécié `oneOf`, 6 `parseString`, 6 `resetCache` dans `_fontconfig_pattern.py`, et 1 `enablePackrat` dans `_mathtext.py`. Il s’agit d’appels internes de dépendance vers d’anciens alias Pyparsing; aucun de ces warnings ne signale une convergence ou un résultat financier invalide. Ils sont documentés et conservés visibles, sans filtre global.

### Exécution et sorties des notebooks

Le kernel `covar-validation` a été enregistré sous le venv. Les commandes suivantes ont été exécutées dans l’ordre. Le premier essai du notebook 01, sans `PYTHONPATH`, a échoué sur `ModuleNotFoundError: No module named 'src'`; aucun code métier n’a été modifié. L’exécution a ensuite réussi en exportant temporairement `PYTHONPATH` vers la racine du projet.

```powershell
$env:PYTHONPATH = 'C:\Users\safaa\OneDrive\Desktop\Projet_ERP'
C:\Users\safaa\.copilot\session-state\485ed53d-7f51-413b-921a-ebaeebec9b88\covar-validation-venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=covar-validation --ExecutePreprocessor.timeout=180 C:\Users\safaa\OneDrive\Desktop\Projet_ERP\notebooks\01_data_preparation.ipynb
C:\Users\safaa\.copilot\session-state\485ed53d-7f51-413b-921a-ebaeebec9b88\covar-validation-venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=covar-validation --ExecutePreprocessor.timeout=300 C:\Users\safaa\OneDrive\Desktop\Projet_ERP\notebooks\02_garch_model.ipynb
C:\Users\safaa\.copilot\session-state\485ed53d-7f51-413b-921a-ebaeebec9b88\covar-validation-venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=covar-validation --ExecutePreprocessor.timeout=600 C:\Users\safaa\OneDrive\Desktop\Projet_ERP\notebooks\03_dcc_garch.ipynb
C:\Users\safaa\.copilot\session-state\485ed53d-7f-51-413b-921a-ebaeebec9b88\covar-validation-venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=covar-validation --ExecutePreprocessor.timeout=900 C:\Users\safaa\OneDrive\Desktop\Projet_ERP\notebooks\04_var_covar.ipynb
```

| Notebook | Statut observé | Sorties vérifiées |
|---|---|---|
| `01_data_preparation.ipynb` | PASS — 2/2 cellules exécutées, aucune erreur | Aperçu et statistiques des 5 séries, 1 304 observations, période de stress, 4 rendus graphiques. |
| `02_garch_model.ipynb` | PASS — 2/2 cellules exécutées, aucune erreur | Les cinq estimations GARCH(1,1) ont convergé et sont déclarées admissibles; volatilités et résidus standardisés affichés, graphiques présents. |
| `03_dcc_garch.ipynb` | PASS — 2/2 cellules exécutées, aucune erreur | `a=0.0`, `b=0.7322178209`, `a+b=0.7322178209 < 1`; 1 304 dates communes gardées; figures de corrélation affichées. |
| `04_var_covar.ipynb` | PASS — 2/2 cellules exécutées, aucune erreur | VaR historique et paramétrique, CoVaR de détresse, CoVaR médian, ΔCoVaR pour les cinq banques, série glissante et graphique. |

Dans le notebook 04, les ΔCoVaR synthétiques par banque sont : ATW_SIM `-0.011406`, BCP_SIM `-0.012355`, BOA_SIM `-0.013471`, CIH_SIM `-0.011874`, CDM_SIM `-0.012398`. Ils sont exprimés en rendement journalier et ne sont pas des estimations empiriques du risque marocain.

Nbconvert a signalé pour chaque notebook un `MissingIDFieldWarning` (cellules historiques sans identifiant explicite), ainsi qu’un warning d’environnement Windows ZMQ/Proactor : un thread sélecteur de compatibilité est utilisé. Ces avertissements n’ont pas empêché l’exécution; aucune cellule ne contient un output de type `error`. Aucun warning de convergence GARCH/DCC n’a interrompu les notebooks.

### Fichiers modifiés ou créés pendant cette phase

**Code, tests et documentation :**

- `src/covar.py`
- `src/preprocessing.py`
- `src/var.py`
- `tests/test_covar.py`
- `tests/test_garch.py`
- `tests/test_var.py`
- `docs/python_guide.md`
- `docs/methodology.md`
- `docs/python_covar_validation_report.md`

**Notebooks exécutés et sauvegardés avec leurs sorties :**

- `notebooks/01_data_preparation.ipynb`
- `notebooks/02_garch_model.ipynb`
- `notebooks/03_dcc_garch.ipynb`
- `notebooks/04_var_covar.ipynb`

**CSV synthétiques créés explicitement :**

- `data/synthetic/bank_prices.csv`
- `data/synthetic/bank_returns.csv`
- `data/synthetic/market_index.csv`
- `data/synthetic/financial_dataset.csv`

### Contrôles de périmètre et limites restantes

- `docker-compose.yml` est inchangé : SHA-256 avant/après `D9A2F74C3CDE8BF0305382D25AB44878E95D930AC9AA59FD287DE3090DA4AB35`. Docker Engine reste indisponible sur le contexte `desktop-linux` en raison du HTTP 500 déjà constaté; les validations ont donc été exécutées dans le venv.
- L’addon `odoo_addons` n’a pas été modifié et contient toujours zéro fichier. Aucun code Odoo, Autoencoder ou EWS n’a été ajouté.
- `requirements.txt` est inchangé. Les dépendances de test de couverture et le kernel Jupyter ont été installés/enregistrés dans le venv externe au projet.
- Les fichiers CSV existants avant cette phase ont été préservés; les quatre CSV créés sont synthétiques, étiquetés, reproductibles et présents comme fichiers canoniques.
- Limitation méthodologique : le calendrier généré est un calendrier de jours ouvrés générique et ne modélise pas les jours fériés marocains. Les valeurs sont simulées, et les sorties des modèles ne doivent pas être présentées comme des résultats financiers réels.
- Il reste des lignes de code non couvertes dans les chemins d’erreur et branches optionnelles des modules; la couverture globale observée est 84 %, sans que cela annule le succès des 52 cas.
