"""SHAP-based explanations and human-readable reason codes.

Two SHAP back-ends are supported:

* **XGBoost** - the Booster's built-in TreeSHAP (``pred_contribs=True``), the
  same exact algorithm as ``shap.TreeExplainer``; values are in log-odds and
  sum to the model output. No extra dependency, no JIT warm-up.
* **scikit-learn tree ensembles** (Random Forest) - ``shap.TreeExplainer``
  from the ``shap`` package; values are in probability space.

``make_explainer`` picks the right one for the deployed model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb


# --------------------------------------------------------------------------- #
# SHAP values
# --------------------------------------------------------------------------- #
def compute_shap_values(booster: xgb.Booster, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Return (shap_values[n, n_features], base_value) in log-odds space."""
    dm = xgb.DMatrix(X, feature_names=list(X.columns))
    contribs = booster.predict(dm, pred_contribs=True)
    shap_values = contribs[:, :-1]
    base_value = float(contribs[0, -1]) if len(contribs) else 0.0
    return shap_values, base_value


class XGBoostExplainer:
    """Native TreeSHAP over a raw ``xgb.Booster`` (log-odds units)."""

    units = "log-odds"

    def __init__(self, booster: xgb.Booster):
        self.booster = booster

    def shap_values(self, X: pd.DataFrame) -> tuple[np.ndarray, float]:
        return compute_shap_values(self.booster, X)


class SklearnTreeExplainer:
    """``shap.TreeExplainer`` over a scikit-learn tree ensemble (probability units)."""

    units = "probability"

    def __init__(self, model):
        import shap  # optional dependency, imported lazily

        self._explainer = shap.TreeExplainer(model)

    def shap_values(self, X: pd.DataFrame) -> tuple[np.ndarray, float]:
        values = self._explainer.shap_values(X)
        if isinstance(values, list):  # older shap: [class0, class1]
            values = values[1]
        elif getattr(values, "ndim", 2) == 3:  # newer shap: (n, features, classes)
            values = values[..., 1]
        base = self._explainer.expected_value
        base = float(np.ravel(base)[-1]) if np.ndim(base) else float(base)
        return np.asarray(values, dtype=float), base


def make_explainer(model_type: str, model, booster: xgb.Booster | None = None):
    """Return an explainer for the deployed model, or ``None`` if unsupported.

    XGBoost is always explainable (native). Random forests need the ``shap``
    package; logistic regression is not explained with SHAP here.
    """
    if model_type == "xgboost" and booster is not None:
        return XGBoostExplainer(booster)
    if model_type == "random_forest":
        try:
            return SklearnTreeExplainer(model)
        except ImportError:
            return None
    return None


def mean_abs_shap(explainer_or_booster, X: pd.DataFrame) -> pd.Series:
    """Global feature importance = mean |SHAP| over the rows of X."""
    if isinstance(explainer_or_booster, xgb.Booster):
        values, _ = compute_shap_values(explainer_or_booster, X)
    else:
        values, _ = explainer_or_booster.shap_values(X)
    return pd.Series(np.abs(values).mean(axis=0), index=list(X.columns)).sort_values(
        ascending=False
    )


# --------------------------------------------------------------------------- #
# Per-transaction explanation
# --------------------------------------------------------------------------- #
@dataclass
class FeatureContribution:
    feature: str
    value: float
    shap_value: float
    direction: str  # "increases_fraud_risk" | "decreases_fraud_risk"

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "value": self.value,
            "shap_value": round(self.shap_value, 4),
            "direction": self.direction,
        }


def top_contributions(
    feature_names: list[str],
    feature_values: np.ndarray,
    shap_row: np.ndarray,
    top_n: int = 5,
) -> list[FeatureContribution]:
    """The ``top_n`` features with the largest absolute SHAP value for one row."""
    order = np.argsort(-np.abs(shap_row))[:top_n]
    out = []
    for i in order:
        s = float(shap_row[i])
        if s == 0.0:
            continue
        v = feature_values[i]
        out.append(
            FeatureContribution(
                feature=str(feature_names[i]),
                value=float(v) if np.isfinite(float(v)) else 0.0,
                shap_value=s,
                direction="increases_fraud_risk" if s > 0 else "decreases_fraud_risk",
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Reason codes
# --------------------------------------------------------------------------- #
_REASON_TEMPLATES: dict[str, tuple[str, str]] = {
    # feature: (message when it INCREASES risk, message when it DECREASES risk)
    "orig_emptied": (
        "Origin account was completely emptied by this transaction",
        "Origin account was not emptied",
    ),
    "amount_equals_orig_balance": (
        "Amount equals the full origin balance",
        "Amount differs from the origin balance",
    ),
    "error_balance_orig": (
        "Origin balances do not reconcile with the amount",
        "Origin balances reconcile with the amount",
    ),
    "error_balance_dest": (
        "Destination balances do not reconcile with the amount",
        "Destination balances reconcile with the amount",
    ),
    "dest_zero_before": (
        "Destination account had a zero balance before the transaction",
        "Destination account already held funds",
    ),
    "dest_zero_after": (
        "Destination balance is zero after receiving funds",
        "Destination balance reflects the received funds",
    ),
    "orig_zero_after": (
        "Origin balance is zero after the transaction",
        "Origin account retains a balance",
    ),
    "orig_zero_before": (
        "Origin account had no funds before the transaction",
        "Origin account held funds before the transaction",
    ),
    "type_TRANSFER": (
        "Transaction is a TRANSFER (high-risk type)",
        "Transaction is not a TRANSFER",
    ),
    "type_CASH_OUT": (
        "Transaction is a CASH_OUT (high-risk type)",
        "Transaction is not a CASH_OUT",
    ),
    "type_PAYMENT": ("Transaction is a PAYMENT", "Transaction is not a PAYMENT (a low-risk type)"),
    "type_CASH_IN": ("Transaction is a CASH_IN", "Transaction is not a CASH_IN (a low-risk type)"),
    "type_DEBIT": ("Transaction is a DEBIT", "Transaction is not a DEBIT"),
    "is_transfer_or_cashout": (
        "Transaction type is TRANSFER or CASH_OUT - the only types where fraud occurs",
        "Transaction type is one where fraud has not been observed",
    ),
    "amount": ("Transaction amount is unusually large", "Transaction amount is typical"),
    "log_amount": ("Transaction amount is unusually large", "Transaction amount is typical"),
    "amount_to_orig_balance_ratio": (
        "Amount is large relative to the origin balance",
        "Amount is small relative to the origin balance",
    ),
    "amount_to_dest_balance_ratio": (
        "Amount is large relative to the destination balance",
        "Amount is small relative to the destination balance",
    ),
    "orig_balance_delta": ("Large drop in origin balance", "Origin balance change is typical"),
    "dest_balance_delta": (
        "Destination balance change is inconsistent with the amount",
        "Destination balance change is consistent",
    ),
    "oldbalanceOrg": (
        "Origin balance before the transaction is a risk signal",
        "Origin balance before the transaction looks normal",
    ),
    "newbalanceOrig": (
        "Origin balance after the transaction is a risk signal",
        "Origin balance after the transaction looks normal",
    ),
    "oldbalanceDest": (
        "Destination balance before the transaction is a risk signal",
        "Destination balance before the transaction looks normal",
    ),
    "newbalanceDest": (
        "Destination balance after the transaction is a risk signal",
        "Destination balance after the transaction looks normal",
    ),
    "hour_of_day": (
        "Time of day is associated with higher fraud rates",
        "Time of day is associated with lower fraud rates",
    ),
    "day_of_week": (
        "Day of week is associated with higher fraud rates",
        "Day of week is associated with lower fraud rates",
    ),
}


def reason_codes(
    contributions: list[FeatureContribution], max_reasons: int = 3, risk_only: bool = True
) -> list[str]:
    """Short human-readable reasons derived from the top SHAP contributions.

    By default only the factors that *increase* fraud risk are returned,
    because those are what an analyst reviewing a flagged transaction needs.
    """
    reasons: list[str] = []
    for c in contributions:
        if risk_only and c.direction != "increases_fraud_risk":
            continue
        up, down = _REASON_TEMPLATES.get(
            c.feature, (f"{c.feature} increases fraud risk", f"{c.feature} decreases fraud risk")
        )
        reasons.append(up if c.direction == "increases_fraud_risk" else down)
        if len(reasons) >= max_reasons:
            break
    return reasons
