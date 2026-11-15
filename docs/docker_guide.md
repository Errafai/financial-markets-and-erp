# Environnement Python avec Docker

Docker Desktop avec le moteur Linux et Docker Compose doivent être disponibles. L'environnement ne requiert aucune installation locale de Python.

1. Copier `.env.example` vers `.env` s'il n'existe pas encore : `if (-not (Test-Path .env)) { Copy-Item .env.example .env }`.
2. Définir `POSTGRES_PASSWORD` et `JUPYTER_TOKEN` avec des valeurs privées non vides. Ne jamais publier `.env`.
3. Construire et démarrer les services partagés Python/Jupyter, PostgreSQL et Odoo : `docker compose up --build -d`.
4. Vérifier les conteneurs : `docker compose ps`; consulter les journaux : `docker compose logs -f python jupyter db odoo`.
5. Ouvrir JupyterLab : `http://localhost:8888`; le service Odoo collectif est accessible sur `http://localhost:8069`.

PostgreSQL stocke ses données dans un volume nommé et Odoo utilise son propre volume persistant. Odoo se connecte au service PostgreSQL par le nom d'hôte Docker `db`; seul le port local de PostgreSQL est publié sur l'interface de boucle locale. Le répertoire `odoo_addons` est monté pour l'équipe, mais reste sans implémentation dans cette contribution. Le service Python génère les CSV canoniques uniquement si le jeu complet manque et reste disponible pour les tests et commandes Python. Le service Jupyter dépend de son contrôle de santé. Les deux services montent le projet sous `/workspace`. Les CSV synthétiques résident dans le répertoire versionné `data/synthetic/`.

Commandes de travail :

```powershell
docker compose exec python pytest -q
docker compose exec python python -m src.synthetic_data --if-missing
docker compose exec python python -m src.synthetic_data --force
docker compose down
```

Le mode `--force` remplace intentionnellement les séries existantes. `docker compose down` arrête les conteneurs sans supprimer les volumes persistants.
