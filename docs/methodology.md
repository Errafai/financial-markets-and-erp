# Méthodologie quantitative — contribution Python

> Toutes les observations utilisées dans le dépôt sont simulées. Les résultats ne mesurent pas le risque réel d'une banque ni du marché marocain.

## Génération et préparation

Le générateur emploie NumPy `default_rng(20251115)` et produit environ cinq années de jours ouvrés simulés pour cinq séries fictives. Des innovations corrélées associent facteur commun et chocs idiosyncratiques. La variance récursive est GARCH-like et deux fenêtres déterministes appliquent une hausse de volatilité et de dépendance commune. Les prix sont obtenus par capitalisation exponentielle, donc restent positifs. `MASI_SIM` est un indice proxy synthétique.

Le rendement logarithmique est `r_t = ln(P_t/P_(t-1))`. Le premier rendement simulé relie le niveau initial de base (100 pour les banques, 1000 pour l'indice) au premier prix daté; une reconstruction de rendements à partir des seules observations datées commence à la seconde date. Les dates dont tous les prix sont manquants sont préservées avant le décalage temporel; le rendement à la date manquante et celui de la date qui la suit restent absents. Aucun rendement multi-séance n'est ainsi présenté comme un rendement journalier. Les jours de fermeture et jours fériés marocains ne sont pas modélisés.

## GARCH(1,1)

`h_t = omega + alpha * epsilon_(t-1)^2 + beta * h_(t-1)`, avec moyenne conditionnelle nulle et innovations normales. Le modèle `arch` ajuste les rendements exprimés en pourcentage; la volatilité est reconvertie à l'unité de rendement. Les résidus standardisés `z_t = epsilon_t/sqrt(h_t)` alimentent l'étape DCC. Les diagnostics retournent convergence, log-vraisemblance marginale, persistance, autocorrélations des résidus et résidus au carré, et Jarque–Bera. Les p-values sont des diagnostics, pas une validation universelle du modèle.

## DCC-GARCH(1,1)

Les résidus standardisés GARCH alignés forment les colonnes `z_t`. `Q_bar = T^-1 sum(z_t z_t')`, `Q_t = (1-a-b)Q_bar + a z_(t-1)z_(t-1)' + bQ_(t-1)` et `R_t = diag(Q_t)^(-1/2)Q_t diag(Q_t)^(-1/2)`. L'optimisation numérique minimise l'objectif gaussien quasi-conditionnel associé aux corrélations, avec `a >= 0`, `b >= 0`, `a+b < 1`. Le terme retourné comme log-vraisemblance est la contribution de corrélation hors constante indépendante des paramètres. Les matrices doivent être finies, symétriques, de diagonale unitaire et définies positives.

Les corrélations décrivent une dépendance conditionnelle, pas la causalité. Le DCC statique ajuste ses paramètres sur l'échantillon fourni; l'évaluation prédictive demande une estimation strictement entraînement/calibration/test.

## VaR, CoVaR et ΔCoVaR

La VaR historique est `-q_alpha(r)`. Sous hypothèse normale, la VaR paramétrique est `-(mu + Phi^-1(alpha)*sigma)`. L'unité est le rendement; une VaR négative peut se produire si le quantile choisi reste en territoire de rendement positif.

Pour un quantile `q`, la régression ajuste `r_system,t = intercept_q + slope_q * r_bank,t + error_t`. Le système utilisé est la moyenne des autres institutions simulées (proxy leave-one-out), afin d'éviter l'inclusion mécanique du rendement bancaire dans sa variable dépendante. Le CoVaR en détresse est la prédiction au quantile inférieur bancaire; le CoVaR de référence est évalué à la médiane. `ΔCoVaR = CoVaR_détresse - CoVaR_médiane`, conservé en unité de rendement, généralement plus négatif en détresse. Le résultat est une association conditionnelle, pas un effet causal.

Une estimation statique emploie toutes les données fournies et ne constitue pas une prévision hors-échantillon. `rolling_covar` ajuste une fenêtre passée finissant à la date de sortie, sans observations postérieures à cette date.

## Reproductibilité et limites

Les quatre CSV canoniques ont un champ `data_status`; les imports de rendements et exports exigent/conservent une identification explicite comme données synthétiques. Le chargement les crée uniquement s'ils sont tous absents. Un jeu partiel est refusé sans écriture afin de protéger les fichiers restants; `force=True` ou `--force` est nécessaire pour remplacer volontairement l'ensemble complet. Les épisodes de stress et paramètres sont construits pour illustrer la méthode; ils n'ont pas d'interprétation historique. Aucune utilisation réglementaire, prévision de crise ou décision financière ne doit être déduite des résultats.

Les ajustements GARCH qui ne convergent pas ou qui ne respectent pas `omega > 0`, `alpha >= 0`, `beta >= 0`, `alpha + beta < 1` sont rejetés avant la construction du panel destiné au DCC. L'exception de modèle préserve les paramètres estimés et les diagnostics; aucune valeur de substitution n'est injectée. Les résidus admissibles sont alignés par intersection de leurs index et seules les dates communes finies sont utilisées pour DCC. Les observations absentes ne sont ni remplacées par zéro ni interpolées.

La VaR historique est calculée avec le quantile linéaire par défaut de NumPy. L'opposé du quantile est une mesure de perte en unités de rendement : elle est positive si le quantile est négatif, mais peut être négative si même le quantile de queue est un rendement positif. La formule n'est pas tronquée à zéro.

Une série exactement constante ne permet pas d'estimer une volatilité empirique et est rejetée pour la VaR paramétrique. Cette vérification utilise l'étendue observée des données afin de ne pas confondre l'arrondi flottant d'un écart-type calculé avec une volatilité réelle. Une volatilité très faible mais issue de valeurs distinctes est conservée.

Pour la régression quantile CoVaR, Statsmodels IRLS est le solveur principal. S'il signale un cycle ou atteint sa limite d'itérations, SciPy HiGHS résout la formulation linéaire équivalente de la perte quantile pinball, avec coefficients non bornés et variables de résidus positifs/négatifs non négatives. La méthode effectivement employée est retournée (`solver`); si les deux solveurs échouent, le calcul échoue explicitement.
