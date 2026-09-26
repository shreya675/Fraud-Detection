"""Feature engineering and the scikit-learn preprocessing pipeline.

Design rules (leakage prevention):
  * Every engineered feature is computed from the *current row only* - no
    aggregates across transactions, so nothing from the future or from the
    test period can leak into a training row.
  * Nothing derived from the target (``isFraud``) or from the post-hoc
    business rule (``isFlaggedFraud``) is used.
  * The preprocessor (one-hot categories, column order) is *fit on the
    training split only* and then applied unchanged to validation/test/API
    input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from fraudguard.config import INPUT_COLUMNS, TRANSACTION_TYPES

# Numeric columns produced by ``engineer_features`` (order matters - it is the
# model's feature order after the one-hot ``type`` block).
ENGINEERED_NUMERIC_FEATURES = [
    "amount",
    "log_amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "orig_balance_delta",
    "dest_balance_delta",
    "error_balance_orig",
    "error_balance_dest",
    "amount_to_orig_balance_ratio",
    "amount_to_dest_balance_ratio",
    "orig_zero_before",
    "orig_zero_after",
    "orig_emptied",
    "dest_zero_before",
    "dest_zero_after",
    "amount_equals_orig_balance",
    "hour_of_day",
    "day_of_week",
    "is_transfer_or_cashout",
]

CATEGORICAL_FEATURE = "type"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new frame with amount- and balance-based features added.

    Works on a single row (API) or millions of rows (training) identically.
    Only the raw ``INPUT_COLUMNS`` are required; any extra columns are
    carried through untouched.
    """
    missing = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()

    amount = out["amount"].astype("float64")
    old_org = out["oldbalanceOrg"].astype("float64")
    new_org = out["newbalanceOrig"].astype("float64")
    old_dst = out["oldbalanceDest"].astype("float64")
    new_dst = out["newbalanceDest"].astype("float64")

    # --- Amount --------------------------------------------------------- #
    out["log_amount"] = np.log1p(amount.clip(lower=0))

    # --- Balance movements ---------------------------------------------- #
    # How much actually left the origin / arrived at the destination
    out["orig_balance_delta"] = old_org - new_org
    out["dest_balance_delta"] = new_dst - old_dst

    # Accounting discrepancies: in a clean transaction these are ~0.
    # PaySim fraud frequently shows a non-zero "error" because the simulated
    # fraudster empties the account regardless of the recorded amount.
    out["error_balance_orig"] = new_org + amount - old_org
    out["error_balance_dest"] = old_dst + amount - new_dst

    # --- Ratios (add 1 to avoid division by zero) ----------------------- #
    out["amount_to_orig_balance_ratio"] = amount / (old_org + 1.0)
    out["amount_to_dest_balance_ratio"] = amount / (old_dst + 1.0)

    # --- Zero-balance flags --------------------------------------------- #
    out["orig_zero_before"] = (old_org == 0).astype("int8")
    out["orig_zero_after"] = (new_org == 0).astype("int8")
    out["orig_emptied"] = ((old_org > 0) & (new_org == 0)).astype("int8")
    out["dest_zero_before"] = (old_dst == 0).astype("int8")
    out["dest_zero_after"] = (new_dst == 0).astype("int8")
    # Fraudsters typically drain the full balance
    out["amount_equals_orig_balance"] = (np.isclose(amount, old_org) & (amount > 0)).astype("int8")

    # --- Time-of-day (step = hours since start of simulation) ----------- #
    # Absolute ``step`` is deliberately NOT used as a feature: the time-aware
    # split means test steps are always larger than training steps, so the
    # model could never generalise from it.
    step = out["step"].astype("int64")
    out["hour_of_day"] = (step % 24).astype("int8")
    out["day_of_week"] = ((step // 24) % 7).astype("int8")

    # --- Transaction type flag ------------------------------------------ #
    out["is_transfer_or_cashout"] = out["type"].isin(["TRANSFER", "CASH_OUT"]).astype("int8")

    return out


def build_preprocessor() -> Pipeline:
    """Feature engineering + one-hot encoding of ``type`` as one sklearn Pipeline.

    Output is a pandas DataFrame whose columns are ``get_feature_names()``.
    """
    encoder = OneHotEncoder(
        categories=[TRANSACTION_TYPES],
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.int8,
    )
    column_transformer = ColumnTransformer(
        transformers=[
            ("type", encoder, [CATEGORICAL_FEATURE]),
            ("num", "passthrough", ENGINEERED_NUMERIC_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    pipeline = Pipeline(
        steps=[
            ("engineer", FunctionTransformer(engineer_features, validate=False)),
            ("encode", column_transformer),
        ]
    )
    pipeline.set_output(transform="pandas")
    return pipeline


def get_feature_names(preprocessor: Pipeline) -> list[str]:
    """Column names produced by a *fitted* preprocessor."""
    return list(preprocessor.named_steps["encode"].get_feature_names_out())


def expected_feature_names() -> list[str]:
    """The feature names the preprocessor will produce (without fitting)."""
    return [f"type_{t}" for t in TRANSACTION_TYPES] + ENGINEERED_NUMERIC_FEATURES
