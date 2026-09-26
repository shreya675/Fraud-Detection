"""Loading, cleaning, and time-aware splitting of the PaySim dataset.

Run as a script to produce the processed splits:

    python -m fraudguard.data

Outputs (data/processed/):
    train.pkl, validation.pkl, test.pkl   - cleaned raw-column frames
    reports/data_summary.json             - row counts, fraud rates, step ranges
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fraudguard.config import (
    DROP_COLUMNS,
    PROCESSED_DATA_DIR,
    RAW_COLUMNS,
    RAW_DATA_FILE,
    REPORTS_DIR,
    TARGET,
    TIME_COL,
    TRAIN_FRACTION,
    TRANSACTION_TYPES,
    VALIDATION_FRACTION,
)

logger = logging.getLogger(__name__)

TRAIN_FILE = PROCESSED_DATA_DIR / "train.pkl"
VALIDATION_FILE = PROCESSED_DATA_DIR / "validation.pkl"
TEST_FILE = PROCESSED_DATA_DIR / "test.pkl"
SUMMARY_FILE = REPORTS_DIR / "data_summary.json"

RAW_DTYPES = {
    "step": "int32",
    "type": "category",
    "amount": "float64",
    "nameOrig": "string",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "string",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_raw(path: Path = RAW_DATA_FILE, nrows: int | None = None) -> pd.DataFrame:
    """Read the raw PaySim CSV with explicit dtypes."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"PaySim CSV not found at {path}. Download it from "
            "https://www.kaggle.com/datasets/ealaxi/paysim1 or run scripts/download_data.py"
        )
    logger.info("Loading %s", path)
    df = pd.read_csv(path, dtype=RAW_DTYPES, nrows=nrows)
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Raw data is missing expected columns: {sorted(missing)}")
    return df


# --------------------------------------------------------------------------- #
# Clean
# --------------------------------------------------------------------------- #
@dataclass
class CleaningReport:
    rows_in: int
    duplicates_removed: int
    rows_missing_target_removed: int
    rows_invalid_type_removed: int
    rows_negative_amount_removed: int
    numeric_nans_filled: int
    rows_out: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Remove duplicates and handle missing / invalid values.

    PaySim is a clean simulated dataset (no NaNs, no duplicates) but the
    function is defensive so the same code path works for real exports.
    """
    rows_in = len(df)
    out = df.drop_duplicates()
    duplicates_removed = rows_in - len(out)

    # Target must be present and binary
    n = len(out)
    if TARGET in out.columns:
        out = out[out[TARGET].notna()]
        out[TARGET] = out[TARGET].astype("int8")
    rows_missing_target_removed = n - len(out)

    # Transaction type must be one of the known categories
    n = len(out)
    out = out[out["type"].astype(str).isin(TRANSACTION_TYPES)]
    rows_invalid_type_removed = n - len(out)
    out["type"] = pd.Categorical(out["type"].astype(str), categories=TRANSACTION_TYPES)

    # Negative amounts are impossible
    n = len(out)
    out = out[out["amount"] >= 0]
    rows_negative_amount_removed = n - len(out)

    # Missing balances -> 0 (PaySim uses 0 for "unknown" merchant balances anyway)
    numeric_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    numeric_nans = int(out[numeric_cols].isna().sum().sum())
    if numeric_nans:
        out[numeric_cols] = out[numeric_cols].fillna(0.0)

    out = out.reset_index(drop=True)
    report = CleaningReport(
        rows_in=rows_in,
        duplicates_removed=duplicates_removed,
        rows_missing_target_removed=rows_missing_target_removed,
        rows_invalid_type_removed=rows_invalid_type_removed,
        rows_negative_amount_removed=rows_negative_amount_removed,
        numeric_nans_filled=numeric_nans,
        rows_out=len(out),
    )
    logger.info("Cleaning report: %s", report)
    return out, report


def drop_non_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ID columns and the post-hoc flag that must never be model inputs."""
    return df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
def time_aware_split(
    df: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split chronologically on ``step`` so the model is always evaluated on
    transactions that happen *after* everything it was trained on.

    Boundaries are chosen on row-count quantiles of ``step`` (so each split
    gets roughly the requested share of rows) and then snapped to whole
    steps, guaranteeing that no single hour is shared between two splits.
    """
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("fractions must be in (0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fractions must leave room for a test set")

    steps = df[TIME_COL].to_numpy()
    train_end = int(np.quantile(steps, train_fraction))
    val_end = int(np.quantile(steps, train_fraction + validation_fraction))
    if not train_end < val_end:
        raise ValueError("Not enough distinct steps to create three non-empty splits")

    train = df[steps <= train_end]
    validation = df[(steps > train_end) & (steps <= val_end)]
    test = df[steps > val_end]

    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        if part.empty:
            raise ValueError(f"{name} split is empty")

    # Sanity: strictly increasing time ranges
    assert (
        train[TIME_COL].max()
        < validation[TIME_COL].min()
        <= validation[TIME_COL].max()
        < test[TIME_COL].min()
    )
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def describe_split(name: str, df: pd.DataFrame) -> dict:
    return {
        "split": name,
        "rows": int(len(df)),
        "fraud": int(df[TARGET].sum()),
        "fraud_rate_pct": round(float(df[TARGET].mean() * 100), 4),
        "step_min": int(df[TIME_COL].min()),
        "step_max": int(df[TIME_COL].max()),
    }


# --------------------------------------------------------------------------- #
# Script entry point
# --------------------------------------------------------------------------- #
def prepare_datasets(raw_path: Path = RAW_DATA_FILE, nrows: int | None = None) -> dict:
    """Full pipeline: load -> clean -> drop non-features -> time split -> save."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw(raw_path, nrows=nrows)
    cleaned, cleaning_report = clean(raw)
    del raw

    dataset_overview = {
        "rows": int(len(cleaned)),
        "fraud": int(cleaned[TARGET].sum()),
        "fraud_rate_pct": round(float(cleaned[TARGET].mean() * 100), 4),
        "steps": [int(cleaned[TIME_COL].min()), int(cleaned[TIME_COL].max())],
        "fraud_by_type": {
            t: int(v) for t, v in cleaned.groupby("type", observed=True)[TARGET].sum().items()
        },
        "rows_by_type": {t: int(v) for t, v in cleaned["type"].value_counts().items()},
        "isFlaggedFraud_total": int(cleaned["isFlaggedFraud"].sum()),
    }

    model_frame = drop_non_features(cleaned)
    del cleaned
    train, validation, test = time_aware_split(model_frame)
    del model_frame

    train.to_pickle(TRAIN_FILE)
    validation.to_pickle(VALIDATION_FILE)
    test.to_pickle(TEST_FILE)

    summary = {
        "source_file": str(raw_path),
        "cleaning": cleaning_report.to_dict(),
        "dataset": dataset_overview,
        "splits": [
            describe_split("train", train),
            describe_split("validation", validation),
            describe_split("test", test),
        ],
        "split_strategy": (
            f"time-aware on '{TIME_COL}': first {TRAIN_FRACTION:.0%} of rows by step -> train, "
            f"next {VALIDATION_FRACTION:.0%} -> validation, remaining -> untouched test"
        ),
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2))
    logger.info("Saved splits to %s and summary to %s", PROCESSED_DATA_DIR, SUMMARY_FILE)
    return summary


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the processed splits written by ``prepare_datasets``."""
    for f in (TRAIN_FILE, VALIDATION_FILE, TEST_FILE):
        if not f.exists():
            raise FileNotFoundError(f"{f} not found - run `python -m fraudguard.data` first")
    return pd.read_pickle(TRAIN_FILE), pd.read_pickle(VALIDATION_FILE), pd.read_pickle(TEST_FILE)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    nrows_arg = int(sys.argv[1]) if len(sys.argv) > 1 else None
    result = prepare_datasets(nrows=nrows_arg)
    print(json.dumps(result, indent=2))
