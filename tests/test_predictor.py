import json

import numpy as np
import pandas as pd
import pytest

from fraudguard.config import TARGET
from fraudguard.explain import FeatureContribution, reason_codes
from fraudguard.predictor import InputValidationError, risk_level, validate_transactions


# ---- training artifacts --------------------------------------------------- #
def test_training_writes_all_artifacts(trained_artifacts, tmp_root):
    models = tmp_root / "models"
    for f in ("preprocessor.joblib", "threshold.json", "feature_names.json", "model_metadata.json"):
        assert (models / f).exists(), f
    assert (models / trained_artifacts["model_file"]).exists()
    reports = tmp_root / "reports"
    for f in (
        "model_comparison.json",
        "model_comparison.csv",
        "threshold_analysis.csv",
        "test_metrics.json",
    ):
        assert (reports / f).exists(), f
    comparison = json.loads((reports / "model_comparison.json").read_text())
    assert set(comparison["models"]) == {"logistic_regression", "random_forest", "xgboost"}
    assert comparison["selected_model"] == trained_artifacts["model_type"]


def test_threshold_was_selected_on_validation_only(trained_artifacts):
    t = trained_artifacts["threshold"]
    assert 0 < t["threshold"] < 1
    assert "validation" in t["rule"]
    # test metrics exist and are internally consistent
    tm = trained_artifacts["test_metrics"]
    assert tm["tp"] + tm["fn"] == tm["positives"]
    assert tm["tp"] + tm["fp"] + tm["fn"] + tm["tn"] == tm["n"]


# ---- predictor ------------------------------------------------------------ #
def test_predict_one_returns_probability_and_decision(
    predictor, fraud_like_transaction, legit_like_transaction
):
    fraud = predictor.predict_one(fraud_like_transaction)
    legit = predictor.predict_one(legit_like_transaction)
    assert 0.0 <= fraud.fraud_probability <= 1.0 and 0.0 <= legit.fraud_probability <= 1.0
    assert fraud.fraud_probability > legit.fraud_probability
    assert fraud.is_fraud is True and legit.is_fraud is False
    assert fraud.threshold == predictor.threshold


def test_predict_one_threshold_override(predictor, fraud_like_transaction):
    low = predictor.predict_one(fraud_like_transaction, threshold=0.01)
    high = predictor.predict_one(fraud_like_transaction, threshold=0.999999)
    assert low.is_fraud is True
    assert high.is_fraud is False or high.fraud_probability >= 0.999999
    with pytest.raises(InputValidationError):
        predictor.predict_one(fraud_like_transaction, threshold=1.5)


def test_predict_frame_batch_matches_single(predictor, synthetic_splits):
    _, _, test = synthetic_splits
    batch = test.head(40)
    scored = predictor.predict_frame(batch, explain=False)
    assert len(scored) == 40
    assert {"fraud_probability", "is_fraud", "risk_level"} <= set(scored.columns)
    assert TARGET in scored.columns  # extra columns are carried through
    single = predictor.predict_one(
        batch.iloc[0][
            [
                "step",
                "type",
                "amount",
                "oldbalanceOrg",
                "newbalanceOrig",
                "oldbalanceDest",
                "newbalanceDest",
            ]
        ].to_dict(),
        explain=False,
    )
    assert single.fraud_probability == pytest.approx(
        float(scored["fraud_probability"].iloc[0]), abs=1e-6
    )


def test_predict_frame_quality_on_synthetic_test_split(predictor, synthetic_splits):
    _, _, test = synthetic_splits
    scored = predictor.predict_frame(test, explain=False)
    y = test[TARGET].to_numpy()
    pred = scored["is_fraud"].to_numpy()
    recall = (pred & (y == 1)).sum() / (y == 1).sum()
    precision = (pred & (y == 1)).sum() / max(pred.sum(), 1)
    assert recall > 0.9 and precision > 0.9


def test_explanations_present_when_supported(xgb_predictor, fraud_like_transaction):
    predictor = xgb_predictor
    assert predictor.supports_shap
    res = predictor.predict_one(fraud_like_transaction, explain=True, top_n=5)
    assert res.explanation is not None
    assert 1 <= len(res.explanation.top_features) <= 5
    for f in res.explanation.top_features:
        assert f["feature"] in predictor.feature_names
        assert f["direction"] in ("increases_fraud_risk", "decreases_fraud_risk")
    assert isinstance(res.explanation.reasons, list)


def test_shap_values_sum_to_model_output(xgb_predictor, synthetic_splits):
    """SHAP contributions + base value must reproduce the model's log-odds."""
    predictor = xgb_predictor
    from fraudguard.explain import compute_shap_values

    _, _, test = synthetic_splits
    X = predictor.transform(test.head(20))
    shap_values, base = compute_shap_values(predictor.booster, X)
    logit = shap_values.sum(axis=1) + base
    prob = 1 / (1 + np.exp(-logit))
    assert np.allclose(prob, predictor.model.predict_proba(X)[:, 1], atol=1e-4)


# ---- validation ----------------------------------------------------------- #
def test_validate_transactions_rejects_bad_input():
    good = {
        "step": 1,
        "type": "PAYMENT",
        "amount": 1.0,
        "oldbalanceOrg": 1.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    }
    with pytest.raises(InputValidationError):
        validate_transactions(pd.DataFrame([]))
    with pytest.raises(InputValidationError):
        validate_transactions(pd.DataFrame([{k: v for k, v in good.items() if k != "amount"}]))
    with pytest.raises(InputValidationError):
        validate_transactions(pd.DataFrame([{**good, "type": "WIRE"}]))
    with pytest.raises(InputValidationError):
        validate_transactions(pd.DataFrame([{**good, "amount": -1}]))
    with pytest.raises(InputValidationError):
        validate_transactions(pd.DataFrame([{**good, "amount": "abc"}]))
    out = validate_transactions(pd.DataFrame([{**good, "type": " payment "}]))
    assert out["type"].iloc[0] == "PAYMENT"


def test_risk_level_bands():
    assert risk_level(0.99, 0.5) == "HIGH"
    assert risk_level(0.6, 0.5) == "MEDIUM"
    assert risk_level(0.3, 0.5) == "LOW"
    assert risk_level(0.01, 0.5) == "MINIMAL"


def test_reason_codes_only_risk_increasing_by_default():
    contribs = [
        FeatureContribution("orig_emptied", 1.0, 2.0, "increases_fraud_risk"),
        FeatureContribution("type_PAYMENT", 0.0, -1.0, "decreases_fraud_risk"),
        FeatureContribution("unknown_feature", 3.0, 0.5, "increases_fraud_risk"),
    ]
    reasons = reason_codes(contribs)
    assert reasons[0].startswith("Origin account was completely emptied")
    assert len(reasons) == 2 and "unknown_feature increases fraud risk" in reasons
    assert len(reason_codes(contribs, risk_only=False)) == 3


def test_batch_explanations_have_reasons(xgb_predictor, synthetic_splits):
    _, _, test = synthetic_splits
    scored = xgb_predictor.predict_frame(test.head(30), explain=True, top_n=3)
    assert "reasons" in scored.columns and "top_features" in scored.columns
    assert all(len(t) <= 3 for t in scored["top_features"])
    flagged = scored[scored["is_fraud"]]
    if len(flagged):
        assert any(len(r) > 0 for r in flagged["reasons"])


def test_predictor_missing_artifacts_raises(tmp_path):
    from fraudguard.predictor import FraudPredictor

    with pytest.raises(FileNotFoundError):
        FraudPredictor(models_dir=tmp_path)
