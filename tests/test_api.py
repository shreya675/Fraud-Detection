import io

import pandas as pd
import pytest

from fraudguard.config import INPUT_COLUMNS


# ---- system endpoints ----------------------------------------------------- #
def test_root_redirects_to_docs(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/docs"


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True and body["database"] == "ok"


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_type"] in ("logistic_regression", "random_forest", "xgboost")
    assert body["n_features"] == len(body["feature_names"]) == 26
    assert 0 < body["threshold"]["threshold"] < 1
    assert body["input_columns"] == INPUT_COLUMNS
    assert "test_metrics" in body and "validation_metrics" in body


# ---- /predict ------------------------------------------------------------- #
def test_predict_fraud_like(client, fraud_like_transaction):
    r = client.post("/predict", json={"transaction": fraud_like_transaction})
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["fraud_probability"] <= 1
    assert body["is_fraud"] is True
    assert body["risk_level"] in ("MINIMAL", "LOW", "MEDIUM", "HIGH")
    assert body["prediction_id"] is not None
    if body["explanation"] is not None:
        assert body["explanation"]["top_features"] and isinstance(
            body["explanation"]["reasons"], list
        )


def test_predict_legit_like_and_no_explanation(client, legit_like_transaction):
    r = client.post("/predict", json={"transaction": legit_like_transaction, "explain": False})
    assert r.status_code == 200
    body = r.json()
    assert body["is_fraud"] is False and body["explanation"] is None


def test_predict_threshold_override(client, fraud_like_transaction):
    r = client.post(
        "/predict",
        json={"transaction": fraud_like_transaction, "threshold": 0.999999, "explain": False},
    )
    assert r.status_code == 200
    assert r.json()["threshold"] == pytest.approx(0.999999)


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "TRANSFER"},  # missing fields
        {
            "step": 1,
            "type": "WIRE",
            "amount": 1,
            "oldbalanceOrg": 1,
            "newbalanceOrig": 0,
            "oldbalanceDest": 0,
            "newbalanceDest": 0,
        },  # bad type
        {
            "step": 1,
            "type": "TRANSFER",
            "amount": -5,
            "oldbalanceOrg": 1,
            "newbalanceOrig": 0,
            "oldbalanceDest": 0,
            "newbalanceDest": 0,
        },  # negative
        {
            "step": 1,
            "type": "TRANSFER",
            "amount": "lots",
            "oldbalanceOrg": 1,
            "newbalanceOrig": 0,
            "oldbalanceDest": 0,
            "newbalanceDest": 0,
        },  # non-numeric
    ],
)
def test_predict_invalid_input_returns_422(client, bad):
    r = client.post("/predict", json={"transaction": bad})
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "validation_error" and body["detail"]


def test_predict_invalid_threshold_returns_422(client, fraud_like_transaction):
    r = client.post("/predict", json={"transaction": fraud_like_transaction, "threshold": 2})
    assert r.status_code == 422


def test_predict_lowercase_type_is_normalised(client, fraud_like_transaction):
    r = client.post(
        "/predict",
        json={"transaction": {**fraud_like_transaction, "type": "transfer"}, "explain": False},
    )
    assert r.status_code == 200


# ---- /predict-batch ------------------------------------------------------- #
def _csv_upload(df: pd.DataFrame, name="tx.csv"):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return {"file": (name, buf.getvalue().encode(), "text/csv")}


def test_predict_batch_csv(client, synthetic_splits):
    _, _, test = synthetic_splits
    df = test.head(100)
    r = client.post("/predict-batch", files=_csv_upload(df))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["summary"]["rows"] == 100
    assert len(body["results"]) == 100
    assert body["summary"]["flagged"] == sum(x["is_fraud"] for x in body["results"])
    assert body["summary"]["batch_id"] != "unsaved"
    assert body["results"][0]["row"] == 0
    # ground truth in the CSV is ignored by the model but recall should be high
    y = df["isFraud"].to_numpy()
    pred = [x["is_fraud"] for x in body["results"]]
    tp = sum(1 for a, b in zip(y, pred, strict=True) if a == 1 and b)
    assert tp / max(y.sum(), 1) > 0.9


def test_predict_batch_threshold_and_no_explain(client, synthetic_splits):
    _, _, test = synthetic_splits
    r = client.post(
        "/predict-batch",
        files=_csv_upload(test.head(20)),
        params={"threshold": 0.999999, "explain": "false"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["threshold"] == pytest.approx(0.999999)
    assert all(x["reasons"] == [] for x in body["results"])


def test_predict_batch_rejects_missing_columns(client):
    r = client.post("/predict-batch", files=_csv_upload(pd.DataFrame({"amount": [1, 2]})))
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_input"


def test_predict_batch_rejects_non_csv_and_empty(client):
    r = client.post("/predict-batch", files={"file": ("tx.txt", b"hello", "text/plain")})
    assert r.status_code == 422
    r = client.post("/predict-batch", files=_csv_upload(pd.DataFrame(columns=INPUT_COLUMNS)))
    assert r.status_code == 422


def test_predict_batch_rejects_unknown_type(client, fraud_like_transaction):
    df = pd.DataFrame([{**fraud_like_transaction, "type": "WIRE"}])
    r = client.post("/predict-batch", files=_csv_upload(df))
    assert r.status_code == 422
    assert "Unknown transaction type" in r.json()["detail"]


# ---- history -------------------------------------------------------------- #
def test_history_endpoints(client, fraud_like_transaction):
    client.post("/predict", json={"transaction": fraud_like_transaction})
    r = client.get("/history", params={"limit": 5})
    assert r.status_code == 200
    items = r.json()["items"]
    assert items and {"fraud_probability", "is_fraud", "created_at", "source"} <= set(items[0])
    r = client.get("/history/stats")
    assert r.status_code == 200 and r.json()["total_predictions"] >= 1
    r = client.get("/history", params={"only_fraud": True, "limit": 3})
    assert all(i["is_fraud"] for i in r.json()["items"])
