"""Central configuration: paths, column names, and training constants.

Every other module imports from here so that a path or column name is
defined exactly once.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(os.environ.get("FRAUDGUARD_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "samples"
MODELS_DIR = Path(os.environ.get("FRAUDGUARD_MODELS_DIR", PROJECT_ROOT / "models"))
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Kaggle PaySim file name: https://www.kaggle.com/datasets/ealaxi/paysim1
RAW_DATA_FILE = RAW_DATA_DIR / "PS_20174392719_1491204439457_log.csv"

# Saved artifacts
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.joblib"
MODEL_FILE = MODELS_DIR / "xgboost_model.ubj"
THRESHOLD_FILE = MODELS_DIR / "threshold.json"
FEATURE_NAMES_FILE = MODELS_DIR / "feature_names.json"
METADATA_FILE = MODELS_DIR / "model_metadata.json"

# SQLite prediction history (override with env var in Docker)
DATABASE_URL = os.environ.get(
    "FRAUDGUARD_DATABASE_URL", f"sqlite:///{PROJECT_ROOT / 'fraudguard.db'}"
)

# --------------------------------------------------------------------------- #
# Dataset schema (PaySim)
# --------------------------------------------------------------------------- #
TARGET = "isFraud"
TIME_COL = "step"  # 1 step == 1 hour of simulated time, 744 steps == 31 days

# Raw columns exactly as they appear in the Kaggle CSV
RAW_COLUMNS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]

# Columns the API accepts for a single transaction (what a real-time scorer would see)
INPUT_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

# Columns dropped before modelling and why:
#   nameOrig / nameDest -> high-cardinality IDs, pure noise for a tree model, leak nothing useful
#   isFlaggedFraud      -> a downstream business rule (amount > 200k), not an input feature
#   isFraud             -> the target
DROP_COLUMNS = ["nameOrig", "nameDest", "isFlaggedFraud"]

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

# --------------------------------------------------------------------------- #
# Split & training constants
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
# Time-aware split on `step`: first 60% of hours -> train, next 20% -> validation, last 20% -> test
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
# The remaining 0.20 is the untouched test set.

DEFAULT_THRESHOLD = 0.5
