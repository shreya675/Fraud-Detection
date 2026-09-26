import numpy as np
import pandas as pd
import pytest

from fraudguard.config import DROP_COLUMNS, INPUT_COLUMNS, TARGET, TRANSACTION_TYPES
from fraudguard.data import clean, drop_non_features, time_aware_split
from fraudguard.features import (
    ENGINEERED_NUMERIC_FEATURES,
    build_preprocessor,
    engineer_features,
    expected_feature_names,
    get_feature_names,
)


# ---- cleaning ------------------------------------------------------------- #
def test_clean_removes_duplicates_and_invalid_rows(synthetic_raw):
    df = synthetic_raw.copy()
    df = pd.concat([df, df.head(10)], ignore_index=True)  # duplicates
    df.loc[20, "amount"] = -5  # negative amount (row 20 is not one of the duplicated rows)
    df["type"] = df["type"].astype(str)
    df.loc[21, "type"] = "BOGUS"
    cleaned, report = clean(df)
    assert report.duplicates_removed == 10
    assert report.rows_negative_amount_removed == 1
    assert report.rows_invalid_type_removed == 1
    assert len(cleaned) == len(synthetic_raw) - 2
    assert set(cleaned["type"].astype(str)) <= set(TRANSACTION_TYPES)


def test_clean_fills_numeric_nans(synthetic_raw):
    df = synthetic_raw.copy()
    df.loc[5, "oldbalanceDest"] = np.nan
    cleaned, report = clean(df)
    assert report.numeric_nans_filled == 1
    assert cleaned["oldbalanceDest"].isna().sum() == 0


def test_drop_non_features_removes_ids_and_flag(synthetic_raw):
    out = drop_non_features(synthetic_raw)
    for c in DROP_COLUMNS:
        assert c not in out.columns
    assert TARGET in out.columns


# ---- split ---------------------------------------------------------------- #
def test_time_split_is_chronological_and_disjoint(synthetic_splits):
    train, val, test = synthetic_splits
    assert train["step"].max() < val["step"].min()
    assert val["step"].max() < test["step"].min()
    assert len(train) > len(val) and len(train) > len(test)
    assert train[TARGET].sum() > 0 and val[TARGET].sum() > 0 and test[TARGET].sum() > 0


def test_time_split_rejects_bad_fractions(synthetic_raw):
    with pytest.raises(ValueError):
        time_aware_split(synthetic_raw, train_fraction=0.7, validation_fraction=0.4)


# ---- features ------------------------------------------------------------- #
def test_engineer_features_adds_expected_columns(synthetic_raw):
    out = engineer_features(synthetic_raw.head(50))
    for c in ENGINEERED_NUMERIC_FEATURES:
        assert c in out.columns, c
    assert out["log_amount"].ge(0).all()
    assert out["hour_of_day"].between(0, 23).all()
    assert out["day_of_week"].between(0, 6).all()
    assert (
        out["is_transfer_or_cashout"]
        == out["type"].astype(str).isin(["TRANSFER", "CASH_OUT"]).astype(int)
    ).all()


def test_engineer_features_balance_error_logic():
    row = pd.DataFrame(
        [
            {
                "step": 1,
                "type": "TRANSFER",
                "amount": 100.0,
                "oldbalanceOrg": 100.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
            }
        ]
    )
    out = engineer_features(row).iloc[0]
    assert out["error_balance_orig"] == 0.0  # 0 + 100 - 100
    assert out["error_balance_dest"] == 100.0  # 0 + 100 - 0 -> destination not credited
    assert out["orig_emptied"] == 1
    assert out["amount_equals_orig_balance"] == 1
    assert out["dest_zero_before"] == 1 and out["dest_zero_after"] == 1


def test_engineer_features_requires_input_columns():
    with pytest.raises(ValueError):
        engineer_features(pd.DataFrame({"amount": [1.0]}))


def test_engineer_features_does_not_use_target_or_ids(synthetic_raw):
    """Leakage guard: features must be computable from INPUT_COLUMNS alone."""
    only_inputs = synthetic_raw[INPUT_COLUMNS].head(20)
    out = engineer_features(only_inputs)
    assert TARGET not in out.columns
    assert "isFlaggedFraud" not in out.columns


def test_preprocessor_output_matches_expected_names(synthetic_splits):
    train, _, _ = synthetic_splits
    pre = build_preprocessor()
    X = pre.fit_transform(train.drop(columns=[TARGET]))
    assert list(X.columns) == expected_feature_names() == get_feature_names(pre)
    assert X.shape[1] == 26
    assert not X.isna().any().any()


def test_preprocessor_handles_single_row_and_unknown_type(synthetic_splits):
    train, _, _ = synthetic_splits
    pre = build_preprocessor().fit(train.drop(columns=[TARGET]))
    row = pd.DataFrame(
        [
            {
                "step": 7,
                "type": "PAYMENT",
                "amount": 5.0,
                "oldbalanceOrg": 10.0,
                "newbalanceOrig": 5.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
            }
        ]
    )
    X = pre.transform(row)
    assert X.shape == (1, 26)
    assert (
        X["type_PAYMENT"].iloc[0] == 1
        and X[["type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_TRANSFER"]].sum().sum() == 0
    )
