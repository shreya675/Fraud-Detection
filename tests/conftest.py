"""Shared fixtures.

A tiny model is trained once per session from synthetic data into a temp
directory; environment variables point the package's config (and therefore
the API and predictor) at it *before* the package is imported.  Tests
therefore never depend on the real PaySim dataset or on locally trained
artifacts, and run identically in CI.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="fraudguard-test-"))
os.environ["FRAUDGUARD_MODELS_DIR"] = str(_TMP / "models")
os.environ["FRAUDGUARD_DATABASE_URL"] = f"sqlite:///{_TMP / 'test.db'}"

import pandas as pd  # noqa: E402

from fraudguard.data import clean, drop_non_features, time_aware_split  # noqa: E402
from tests.synthetic import make_synthetic_paysim  # noqa: E402


@pytest.fixture(scope="session")
def tmp_root() -> Path:
    return _TMP


@pytest.fixture(scope="session")
def synthetic_raw() -> pd.DataFrame:
    return make_synthetic_paysim(n_rows=6000, seed=0)


@pytest.fixture(scope="session")
def synthetic_splits(synthetic_raw):
    cleaned, _ = clean(synthetic_raw)
    return time_aware_split(drop_non_features(cleaned))


@pytest.fixture(scope="session")
def trained_artifacts(synthetic_splits, tmp_root) -> dict:
    """Train quickly on synthetic data; returns the metadata dict."""
    from fraudguard import train

    meta = train.run(
        splits=synthetic_splits,
        models_dir=tmp_root / "models",
        reports_dir=tmp_root / "reports",
        sample_dir=tmp_root / "samples",
        quick=True,
    )
    return meta


@pytest.fixture(scope="session")
def predictor(trained_artifacts):
    from fraudguard.predictor import FraudPredictor

    return FraudPredictor(models_dir=Path(os.environ["FRAUDGUARD_MODELS_DIR"]))


@pytest.fixture(scope="session")
def client(trained_artifacts):
    from api.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


@pytest.fixture
def fraud_like_transaction() -> dict:
    return {
        "step": 300,
        "type": "TRANSFER",
        "amount": 181.0,
        "oldbalanceOrg": 181.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }


@pytest.fixture
def legit_like_transaction() -> dict:
    return {
        "step": 120,
        "type": "PAYMENT",
        "amount": 9839.64,
        "oldbalanceOrg": 170136.0,
        "newbalanceOrig": 160296.36,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }


@pytest.fixture(scope="session")
def xgb_predictor(synthetic_splits, tmp_root):
    """An XGBoost predictor built explicitly so SHAP paths are always exercised."""
    import json

    import joblib
    import xgboost as xgb

    from fraudguard.config import TARGET
    from fraudguard.features import build_preprocessor, get_feature_names
    from fraudguard.predictor import FraudPredictor

    train, val, _ = synthetic_splits
    pre = build_preprocessor()
    X = pre.fit_transform(train.drop(columns=[TARGET]))
    model = xgb.XGBClassifier(
        n_estimators=40, max_depth=4, learning_rate=0.2, tree_method="hist", random_state=0
    )
    model.fit(X, train[TARGET].to_numpy())

    d = tmp_root / "models_xgb"
    d.mkdir(exist_ok=True)
    joblib.dump(pre, d / "preprocessor.joblib")
    model.get_booster().save_model(str(d / "xgboost_model.ubj"))
    names = get_feature_names(pre)
    (d / "feature_names.json").write_text(json.dumps(names))
    (d / "threshold.json").write_text(json.dumps({"threshold": 0.5, "rule": "fixture"}))
    (d / "model_metadata.json").write_text(
        json.dumps({"model_type": "xgboost", "model_file": "xgboost_model.ubj", "version": "test"})
    )
    return FraudPredictor(models_dir=d)
