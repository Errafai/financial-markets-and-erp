"""Shared project settings and synthetic institution labels."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC_DATA_DIR = PROJECT_ROOT / "data" / "synthetic"
REAL_DATA_DIR = PROJECT_ROOT / "data" / "real"
RANDOM_SEED = 20251115
SYNTHETIC_START_DATE = "2021-01-01"
SYNTHETIC_END_DATE = "2025-12-31"

BANKS = {
    "ATW_SIM": "Attijariwafa Bank (simulated)",
    "BCP_SIM": "Banque Centrale Populaire (simulated)",
    "BOA_SIM": "Bank of Africa (simulated)",
    "CIH_SIM": "CIH Bank (simulated)",
    "CDM_SIM": "Crédit du Maroc (simulated)",
}
MARKET_COLUMN = "MASI_SIM"
SYNTHETIC_LABEL = "SYNTHETIC / SIMULATED DATA — NOT REAL MARKET DATA"
REAL_LABEL = "REAL MARKET DATA - FROM LOCAL CSV IMPORTS"
