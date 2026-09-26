"""Tests for the dashboard's non-UI helpers (no Streamlit required)."""

import io
import json

import pandas as pd
import pytest
from dashboard import utils

from fraudguard.config import INPUT_COLUMNS


def test_read_uploaded_csv_validates_columns():
    good = io.StringIO(utils.template_csv_bytes().decode())
    df = utils.read_uploaded_csv(good)
    assert list(df.columns) == INPUT_COLUMNS and len(df) == len(utils.EXAMPLE_TRANSACTIONS)
    with pytest.raises(ValueError):
        utils.read_uploaded_csv(io.StringIO("amount\n1\n"))


def test_scored_to_csv_flattens_reasons():
    df = pd.DataFrame([{"amount": 1.0, "reasons": ["a", "b"]}, {"amount": 2.0, "reasons": []}])
    text = utils.scored_to_csv_bytes(df).decode()
    assert "a | b" in text


def test_comparison_table_from_reports(trained_artifacts, tmp_root):
    comp = json.loads((tmp_root / "reports" / "model_comparison.json").read_text())
    table = utils.comparison_table(comp)
    assert set(table.index) == {"logistic_regression", "random_forest", "xgboost"}
    assert {
        "precision",
        "recall",
        "f1",
        "pr_auc",
        "roc_auc",
        "false_positives",
        "false_negatives",
    } <= set(table.columns)


def test_local_backend_scores_and_records(trained_artifacts, fraud_like_transaction):
    backend = utils.Backend(prefer_api=False)
    assert backend.mode == "local"
    res = backend.predict_one(fraud_like_transaction, threshold=None)
    assert res["is_fraud"] is True and "model_type" in res
    df = pd.DataFrame([fraud_like_transaction] * 3)
    scored = backend.predict_batch(df, threshold=0.5)
    assert len(scored) == 3 and scored["is_fraud"].all()
    assert backend.history_stats()["total_predictions"] >= 4
    hist = backend.history(limit=2)
    assert len(hist) == 2
