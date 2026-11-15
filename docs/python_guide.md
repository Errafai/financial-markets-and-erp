# Guide des modules Python

> Les jeux de données et sorties sont exclusivement synthétiques et ne représentent pas les établissements financiers réels.

Exécuter les quatre notebooks dans l'ordre depuis JupyterLab. Les modules réutilisables résident sous `src/`; les notebooks sont des explications et pilotes de ces fonctions, pas une deuxième implémentation.

## Données et prétraitement

`src.synthetic_data.generate_synthetic_data()` produit les prix, rendements, indice marché proxy et table combinée avec graine fixe `20251115`. `save_synthetic_data()` enregistre les quatre CSV canoniques. Un jeu complètement absent peut être généré à la première lecture; un jeu partiel est refusé pour protéger les fichiers existants. `force=True` est nécessaire pour régénérer volontairement les CSV. Le chargeur valide leurs dates, institutions attendues et marqueurs de données simulées. L'import de rendements CSV exige les colonnes de rendements numériques, `date` et la marque de provenance exacte.

`calculate_log_returns()` calcule `ln(P_t/P_(t-1))`. Les dates où tous les prix sont absents sont préservées jusqu'au calcul du décalage temporel; les rendements de part et d'autre d'une observation manquante restent absents. Aucune variation sur plusieurs jours n'est ainsi présentée comme un rendement journalier.

## Modèles et interprétation

`fit_garch()` ajuste GARCH(1,1) sur les rendements multipliés par 100 pour la stabilité numérique, et renvoie la volatilité dans l'unité d'origine. Un ajustement doit converger, avoir `omega > 0`, `alpha >= 0`, `beta >= 0` et `alpha + beta < 1`. Un échec lève `GARCHFitError`, qui conserve les paramètres estimés et les diagnostics pour analyse, sans fabriquer de valeurs de remplacement. Les résidus standardisés, la convergence, la persistance, Ljung–Box et Jarque–Bera sont fournis pour le diagnostic. Les sorties de modèle doivent être examinées; elles ne prouvent pas une adéquation parfaite.

`align_standardized_residuals()` effectue une jointure interne sur les dates des résidus standardisés; seules les dates présentes et finies pour toutes les banques sont conservées. Aucune valeur n'est imputée ou interpolée. La fonction refuse les ajustements non convergents/non admissibles et exige au moins 30 dates communes. `fit_dcc()` exige ensuite un DataFrame fini, unique et ordonné. La récursion utilise `Q_bar = T^-1 sum(z_t z_t')`, respecte `a >= 0`, `b >= 0`, `a+b < 1` et renvoie les matrices conditionnelles symétriques normalisées. La log-vraisemblance retournée est la partie de corrélation gaussienne quasi-vraisemblable, hors constante indépendante des paramètres.

La VaR historique utilise le quantile linéaire par défaut de NumPy et en prend l'opposé; le test de l'échantillon avec cinq rendements à -10 %, quinze à 0 % et trente à +10 % vaut donc 1 % au quantile 10 %, par interpolation. La VaR est positive lorsque le quantile sélectionné est une perte; si le quantile reste un gain, la formule conserve une VaR négative au lieu de la tronquer artificiellement. Une série strictement constante est rejetée comme variance nulle; une faible volatilité observée mais non nulle reste admise.

CoVaR ajuste une régression quantile entre le rendement de l'institution et celui de son système de pairs; le pair système exclut l'institution étudiée. La méthode principale est l'IRLS de Statsmodels. Si ce solveur signale un cycle ou sa limite d'itérations, une programmation linéaire SciPy HiGHS minimise la même perte quantile pinball; le champ `solver` identifie la méthode retenue et un échec des deux méthodes reste une erreur explicite. `ΔCoVaR` est la différence entre le CoVaR de détresse (quantile inférieur) et la référence médiane. Le classement fondé sur cette valeur est descriptif et ne prouve pas une contribution causale.

## Tests et exports

Les tests sont dans `tests/` et s'exécutent dans le conteneur Python : `docker compose exec python pytest -q`. Les exports CSV/JSON ajoutent `data_status` afin que la nature simulée reste visible au-delà de l'analyse.
