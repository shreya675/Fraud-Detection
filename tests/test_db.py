import pandas as pd

from fraudguard.db import PredictionStore


def _scored_frame(n=3):
    return pd.DataFrame(
        [
            {
                "step": i,
                "type": "TRANSFER",
                "amount": 100.0 * (i + 1),
                "oldbalanceOrg": 100.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
                "fraud_probability": 0.9 if i % 2 == 0 else 0.1,
                "is_fraud": i % 2 == 0,
                "threshold": 0.5,
                "risk_level": "HIGH" if i % 2 == 0 else "LOW",
                "reasons": ["Origin account was completely emptied by this transaction"]
                if i % 2 == 0
                else [],
            }
            for i in range(n)
        ]
    )


def test_store_roundtrip_and_stats():
    store = PredictionStore("sqlite:///:memory:")
    batch_id = store.save_frame(_scored_frame(5), source="test", model_version="v1")
    assert batch_id
    rows = store.recent(limit=10)
    assert len(rows) == 5
    assert rows[0]["reasons"] in ([], ["Origin account was completely emptied by this transaction"])
    assert all(r["batch_id"] == batch_id for r in rows)
    fraud_only = store.recent(limit=10, only_fraud=True)
    assert len(fraud_only) == 3 and all(r["is_fraud"] for r in fraud_only)
    stats = store.stats()
    assert stats["total_predictions"] == 5 and stats["flagged_fraud"] == 3 and stats["batches"] == 1
    assert 0 < stats["flag_rate"] < 1
    assert store.clear() == 5 and store.stats()["total_predictions"] == 0


def test_recent_frame_returns_dataframe():
    store = PredictionStore("sqlite:///:memory:")
    assert store.recent_frame().empty
    store.save_frame(_scored_frame(2), source="test")
    df = store.recent_frame(limit=5)
    assert len(df) == 2 and "fraud_probability" in df.columns
