"""Inference: load saved artifacts and score transactions with explanations.

The same ``FraudPredictor`` is used by the FastAPI service, the Streamlit
dashboard, and the tests, so single and batch scoring share one code path.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from fraudguard.config import (
    FEATURE_NAMES_FILE,
    INPUT_COLUMNS,
    METADATA_FILE,
    MODELS_DIR,
    PREPROCESSOR_FILE,
    THRESHOLD_FILE,
    TRANSACTION_TYPES,
)
from fraudguard.explain import make_explainer, reason_codes, top_contributions

logger = logging.getLogger(__name__)


class InputValidationError(ValueError):
    """Raised when transaction data cannot be scored."""


def _load_preprocessor(path: Path):
    """Load the fitted preprocessor; rebuild it if the pickle is incompatible.

    The preprocessor has no learned state beyond the fixed transaction-type
    categories, so a rebuilt one is identical. This keeps artifacts usable
    across scikit-learn versions (trained on one machine, served on another).
    """
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pre = joblib.load(path)
        # make sure the unpickled object actually works
        pre.transform(
            pd.DataFrame(
                [
                    {
                        "step": 1,
                        "type": "PAYMENT",
                        "amount": 1.0,
                        "oldbalanceOrg": 1.0,
                        "newbalanceOrig": 0.0,
                        "oldbalanceDest": 0.0,
                        "newbalanceDest": 0.0,
                    }
                ]
            )
        )
        return pre
    except Exception as exc:  # version mismatch, corrupt pickle, ...
        logger.warning("Could not use saved preprocessor (%s); rebuilding an equivalent one", exc)
        from fraudguard.features import build_preprocessor

        rows = [
            {
                "step": i,
                "type": t,
                "amount": 1.0,
                "oldbalanceOrg": 1.0,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 0.0,
                "newbalanceDest": 0.0,
            }
            for i, t in enumerate(TRANSACTION_TYPES)
        ]
        return build_preprocessor().fit(pd.DataFrame(rows))


class BoosterClassifier:
    """Minimal sklearn-like ``predict_proba`` over a raw ``xgb.Booster``.

    Works with any xgboost >= 1.4 regardless of the installed scikit-learn.
    """

    def __init__(self, booster: xgb.Booster):
        self.booster = booster

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        dm = xgb.DMatrix(X, feature_names=list(X.columns))
        p = self.booster.predict(dm).astype(float)
        return np.column_stack([1.0 - p, p])


@dataclass
class Explanation:
    top_features: list[dict] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    base_value: float = 0.0


@dataclass
class PredictionResult:
    fraud_probability: float
    is_fraud: bool
    threshold: float
    risk_level: str
    explanation: Explanation | None = None

    def to_dict(self) -> dict:
        d = {
            "fraud_probability": self.fraud_probability,
            "is_fraud": self.is_fraud,
            "threshold": self.threshold,
            "risk_level": self.risk_level,
        }
        if self.explanation is not None:
            d["explanation"] = {
                "top_features": self.explanation.top_features,
                "reasons": self.explanation.reasons,
                "base_value": self.explanation.base_value,
            }
        return d


def risk_level(probability: float, threshold: float) -> str:
    """Coarse label for dashboards: LOW / MEDIUM / HIGH."""
    if probability >= max(threshold, 0.9):
        return "HIGH"
    if probability >= threshold:
        return "MEDIUM"
    if probability >= threshold / 2:
        return "LOW"
    return "MINIMAL"


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Check columns, types and ranges. Returns a cleaned copy or raises."""
    if df is None or len(df) == 0:
        raise InputValidationError("No transactions provided")
    missing = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise InputValidationError(
            f"Missing required columns: {missing}. Required: {INPUT_COLUMNS}"
        )

    out = df.copy()
    out["type"] = out["type"].astype(str).str.upper().str.strip()
    bad_types = sorted(set(out["type"]) - set(TRANSACTION_TYPES))
    if bad_types:
        raise InputValidationError(
            f"Unknown transaction type(s): {bad_types}. Allowed: {TRANSACTION_TYPES}"
        )

    numeric = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]
    for c in numeric:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    nan_rows = out[numeric].isna().any(axis=1)
    if nan_rows.any():
        raise InputValidationError(
            f"{int(nan_rows.sum())} row(s) contain missing or non-numeric values in {numeric}"
        )
    if (out[numeric] < 0).any().any():
        raise InputValidationError("step, amount and balances must be non-negative")
    if not np.isfinite(out[numeric].to_numpy(dtype=float)).all():
        raise InputValidationError("Values must be finite")
    out["step"] = out["step"].astype("int64")
    return out


# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #
class FraudPredictor:
    def __init__(self, models_dir: Path = MODELS_DIR):
        models_dir = Path(models_dir)
        self.models_dir = models_dir
        required = [
            models_dir / PREPROCESSOR_FILE.name,
            models_dir / THRESHOLD_FILE.name,
            models_dir / FEATURE_NAMES_FILE.name,
            models_dir / METADATA_FILE.name,
        ]
        for f in required:
            if not f.exists():
                raise FileNotFoundError(
                    f"Model artifact missing: {f}. Run `python -m fraudguard.train` first."
                )

        self.preprocessor = _load_preprocessor(models_dir / PREPROCESSOR_FILE.name)
        self.feature_names: list[str] = json.loads(
            (models_dir / FEATURE_NAMES_FILE.name).read_text()
        )
        self.threshold_info: dict = json.loads((models_dir / THRESHOLD_FILE.name).read_text())
        self.threshold: float = float(self.threshold_info["threshold"])
        self.metadata: dict = json.loads((models_dir / METADATA_FILE.name).read_text())
        self.model_type: str = self.metadata["model_type"]

        model_file = models_dir / self.metadata["model_file"]
        if self.model_type == "xgboost":
            self.booster = xgb.Booster()
            self.booster.load_model(str(model_file))
            self.model = BoosterClassifier(self.booster)
        else:
            self.model = joblib.load(model_file)
            self.booster = None
        self.explainer = make_explainer(self.model_type, self.model, self.booster)
        logger.info(
            "Loaded %s from %s (threshold=%.2f)", self.model_type, model_file, self.threshold
        )

    # ---- helpers -------------------------------------------------------- #
    @property
    def supports_shap(self) -> bool:
        return self.explainer is not None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.preprocessor.transform(df)
        return X[self.feature_names]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = self.transform(validate_transactions(df))
        return self.model.predict_proba(X)[:, 1].astype(float)

    # ---- scoring ------------------------------------------------------- #
    def predict_frame(
        self,
        df: pd.DataFrame,
        threshold: float | None = None,
        explain: bool = False,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """Score a DataFrame. Returns the input columns plus:
        fraud_probability, is_fraud, risk_level and (if explain) top_features/reasons.
        """
        t = self.threshold if threshold is None else float(threshold)
        if not 0.0 <= t <= 1.0:
            raise InputValidationError("threshold must be within [0, 1]")

        clean = validate_transactions(df)
        X = self.transform(clean)
        prob = self.model.predict_proba(X)[:, 1].astype(float)

        result = clean.reset_index(drop=True).copy()
        result["fraud_probability"] = np.round(prob, 6)
        result["is_fraud"] = prob >= t
        result["risk_level"] = [risk_level(p, t) for p in prob]

        if explain and self.supports_shap:
            shap_values, _ = self.explainer.shap_values(X)
            feats = X.to_numpy(dtype=float)
            tops, reasons = [], []
            for i in range(len(X)):
                contribs = top_contributions(
                    self.feature_names, feats[i], shap_values[i], top_n=top_n
                )
                tops.append([c.to_dict() for c in contribs])
                reasons.append(reason_codes(contribs))
            result["top_features"] = tops
            result["reasons"] = reasons
        return result

    def predict_one(
        self,
        transaction: dict,
        threshold: float | None = None,
        explain: bool = True,
        top_n: int = 5,
    ) -> PredictionResult:
        df = pd.DataFrame([transaction])
        scored = self.predict_frame(df, threshold=threshold, explain=explain, top_n=top_n)
        row = scored.iloc[0]
        explanation = None
        if explain and self.supports_shap:
            X = self.transform(validate_transactions(df))
            _, base = self.explainer.shap_values(X)
            explanation = Explanation(
                top_features=list(row["top_features"]),
                reasons=list(row["reasons"]),
                base_value=round(base, 4),
            )
        return PredictionResult(
            fraud_probability=float(row["fraud_probability"]),
            is_fraud=bool(row["is_fraud"]),
            threshold=self.threshold if threshold is None else float(threshold),
            risk_level=str(row["risk_level"]),
            explanation=explanation,
        )

    # ---- info ----------------------------------------------------------- #
    def info(self) -> dict:
        m = self.metadata
        return {
            "project": m.get("project", "FraudGuard"),
            "version": m.get("version"),
            "model_type": self.model_type,
            "trained_at_utc": m.get("trained_at_utc"),
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "input_columns": INPUT_COLUMNS,
            "transaction_types": TRANSACTION_TYPES,
            "threshold": self.threshold_info,
            "validation_metrics": m.get("validation_metrics"),
            "test_metrics": m.get("test_metrics"),
            "split_strategy": m.get("split_strategy"),
            "environment": m.get("environment"),
            "supports_shap": self.supports_shap,
        }


@lru_cache(maxsize=1)
def get_predictor() -> FraudPredictor:
    """Process-wide cached predictor (artifacts are loaded once)."""
    return FraudPredictor()
