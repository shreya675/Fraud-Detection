"""Non-UI helpers for the Streamlit dashboard.

Kept separate from ``app.py`` so they can be unit-tested without Streamlit.
The dashboard talks to the FastAPI service when ``FRAUDGUARD_API_URL`` is
reachable and falls back to in-process scoring otherwise, so it works both in
Docker Compose (API container) and as a standalone ``streamlit run``.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pandas as pd
import requests

from fraudguard.config import INPUT_COLUMNS, MODELS_DIR, REPORTS_DIR

API_URL = os.environ.get("FRAUDGUARD_API_URL", "http://localhost:8000").rstrip("/")

# Chart palette (same as evaluate.py)
PALETTE = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "yellow": "#eda100",
    "red": "#e34948",
    "neutral": "#52514e",
    "grid": "#e6e5e1",
}
MODEL_COLORS = {
    "logistic_regression": PALETTE["blue"],
    "random_forest": PALETTE["orange"],
    "xgboost": PALETTE["aqua"],
}

EXAMPLE_TRANSACTIONS = {
    "Suspicious TRANSFER (account emptied)": {
        "step": 300,
        "type": "TRANSFER",
        "amount": 181.0,
        "oldbalanceOrg": 181.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    },
    "Suspicious CASH_OUT (large, drained)": {
        "step": 400,
        "type": "CASH_OUT",
        "amount": 1_250_000.0,
        "oldbalanceOrg": 1_250_000.0,
        "newbalanceOrig": 0.0,
        "oldbalanceDest": 50_000.0,
        "newbalanceDest": 1_300_000.0,
    },
    "Normal PAYMENT": {
        "step": 120,
        "type": "PAYMENT",
        "amount": 9_839.64,
        "oldbalanceOrg": 170_136.0,
        "newbalanceOrig": 160_296.36,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0,
    },
    "Normal CASH_IN": {
        "step": 50,
        "type": "CASH_IN",
        "amount": 25_000.0,
        "oldbalanceOrg": 100_000.0,
        "newbalanceOrig": 125_000.0,
        "oldbalanceDest": 900_000.0,
        "newbalanceDest": 875_000.0,
    },
}


# --------------------------------------------------------------------------- #
# Backend selection: API first, local predictor as fallback
# --------------------------------------------------------------------------- #
def api_available(timeout: float = 1.5) -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=timeout)
        return r.ok and r.json().get("model_loaded", False)
    except requests.RequestException:
        return False


class Backend:
    """Uniform scoring interface over the REST API or the in-process predictor."""

    def __init__(self, prefer_api: bool = True):
        self.use_api = prefer_api and api_available()
        self._predictor = None
        self._store = None

    @property
    def mode(self) -> str:
        return "api" if self.use_api else "local"

    # lazy local objects
    @property
    def predictor(self):
        if self._predictor is None:
            from fraudguard.predictor import get_predictor

            self._predictor = get_predictor()
        return self._predictor

    @property
    def store(self):
        if self._store is None:
            from fraudguard.db import get_store

            self._store = get_store()
        return self._store

    # ---- calls ---------------------------------------------------------- #
    def model_info(self) -> dict:
        if self.use_api:
            return requests.get(f"{API_URL}/model-info", timeout=10).json()
        return self.predictor.info()

    def predict_one(self, transaction: dict, threshold: float | None, top_n: int = 5) -> dict:
        if self.use_api:
            r = requests.post(
                f"{API_URL}/predict",
                json={
                    "transaction": transaction,
                    "threshold": threshold,
                    "explain": True,
                    "top_n": top_n,
                },
                timeout=30,
            )
            body = r.json()
            if not r.ok:
                raise ValueError(body.get("detail", body))
            return body
        res = self.predictor.predict_one(
            transaction, threshold=threshold, explain=True, top_n=top_n
        ).to_dict()
        scored = pd.DataFrame(
            [{**transaction, **res, "reasons": res.get("explanation", {}).get("reasons", [])}]
        )
        self.store.save_frame(
            scored, source="dashboard", model_version=self.predictor.metadata.get("trained_at_utc")
        )
        return {**res, "model_type": self.predictor.model_type}

    def predict_batch(self, df: pd.DataFrame, threshold: float | None) -> pd.DataFrame:
        """Return the input rows with fraud_probability / is_fraud / risk_level / reasons."""
        if self.use_api:
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            params = {"explain": "true"}
            if threshold is not None:
                params["threshold"] = threshold
            r = requests.post(
                f"{API_URL}/predict-batch",
                files={"file": ("upload.csv", buf.getvalue().encode(), "text/csv")},
                params=params,
                timeout=300,
            )
            body = r.json()
            if not r.ok:
                raise ValueError(body.get("detail", body))
            results = pd.DataFrame(body["results"]).set_index("row")
            out = df.reset_index(drop=True).copy()
            for c in ("fraud_probability", "is_fraud", "risk_level", "reasons"):
                out[c] = results[c].to_numpy()
            out["threshold"] = body["summary"]["threshold"]
            return out
        scored = self.predictor.predict_frame(df, threshold=threshold, explain=True, top_n=3)
        scored["threshold"] = self.predictor.threshold if threshold is None else threshold
        self.store.save_frame(
            scored,
            source="dashboard-batch",
            model_version=self.predictor.metadata.get("trained_at_utc"),
        )
        return scored.drop(columns=["top_features"], errors="ignore")

    def history(self, limit: int = 200, only_fraud: bool = False) -> pd.DataFrame:
        if self.use_api:
            r = requests.get(
                f"{API_URL}/history", params={"limit": limit, "only_fraud": only_fraud}, timeout=30
            )
            return pd.DataFrame(r.json().get("items", []))
        return self.store.recent_frame(limit=limit, only_fraud=only_fraud)

    def history_stats(self) -> dict:
        if self.use_api:
            return requests.get(f"{API_URL}/history/stats", timeout=10).json()
        return self.store.stats()


# --------------------------------------------------------------------------- #
# Reports (produced by fraudguard.train) - read from disk
# --------------------------------------------------------------------------- #
def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def load_reports() -> dict:
    return {
        "comparison": load_json(REPORTS_DIR / "model_comparison.json"),
        "test_metrics": load_json(REPORTS_DIR / "test_metrics.json"),
        "data_summary": load_json(REPORTS_DIR / "data_summary.json"),
        "metadata": load_json(MODELS_DIR / "model_metadata.json"),
        "threshold_sweep": pd.read_csv(REPORTS_DIR / "threshold_analysis.csv")
        if (REPORTS_DIR / "threshold_analysis.csv").exists()
        else None,
        "shap_importance": pd.read_csv(REPORTS_DIR / "shap_feature_importance.csv", index_col=0)
        if (REPORTS_DIR / "shap_feature_importance.csv").exists()
        else None,
        "figures_dir": REPORTS_DIR / "figures",
    }


def comparison_table(comparison: dict, split: str = "test_at_own_threshold") -> pd.DataFrame:
    """Flatten model_comparison.json into a display table."""
    rows = []
    for name, block in comparison["models"].items():
        m = block[split]
        rows.append(
            {
                "model": name,
                "threshold": m["threshold"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "pr_auc": m["pr_auc"],
                "roc_auc": m["roc_auc"],
                "false_positives": m["fp"],
                "false_negatives": m["fn"],
            }
        )
    return pd.DataFrame(rows).set_index("model")


# --------------------------------------------------------------------------- #
# CSV helpers
# --------------------------------------------------------------------------- #
def read_uploaded_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    missing = [c for c in INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return df


def scored_to_csv_bytes(scored: pd.DataFrame) -> bytes:
    out = scored.copy()
    if "reasons" in out.columns:
        out["reasons"] = out["reasons"].apply(
            lambda r: " | ".join(r) if isinstance(r, (list, tuple)) else r
        )
    return out.to_csv(index=False).encode("utf-8")


def template_csv_bytes() -> bytes:
    rows = list(EXAMPLE_TRANSACTIONS.values())
    return pd.DataFrame(rows, columns=INPUT_COLUMNS).to_csv(index=False).encode("utf-8")


# --------------------------------------------------------------------------- #
# Plotly figure builders
# --------------------------------------------------------------------------- #
def _base_layout(fig, title: str, height: int = 380):
    fig.update_layout(
        title={"text": title, "x": 0, "font": {"size": 15, "color": PALETTE["neutral"]}},
        height=height,
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": -0.2},
        font={"color": PALETTE["neutral"]},
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=PALETTE["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=PALETTE["grid"], zeroline=False)
    return fig


def shap_bar_figure(top_features: list[dict], title: str = "Top SHAP contributions"):
    import plotly.graph_objects as go

    feats = list(reversed(top_features))
    names = [f"{f['feature']} = {f['value']:,.4g}" for f in feats]
    vals = [f["shap_value"] for f in feats]
    colors = [PALETTE["red"] if v > 0 else PALETTE["blue"] for v in vals]
    fig = go.Figure(
        go.Bar(
            x=vals,
            y=names,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>SHAP = %{x:.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_color=PALETTE["neutral"], line_width=1)
    fig = _base_layout(fig, title, height=60 + 40 * len(feats))
    fig.update_layout(hovermode="closest", showlegend=False)
    fig.update_xaxes(title="SHAP value (log-odds)  ← lowers risk | raises risk →")
    return fig


def threshold_figure(sweep: pd.DataFrame, chosen: float, current: float | None = None):
    import plotly.graph_objects as go

    fig = go.Figure()
    for col, color in [
        ("precision", PALETTE["blue"]),
        ("recall", PALETTE["orange"]),
        ("f1", PALETTE["aqua"]),
    ]:
        fig.add_trace(
            go.Scatter(
                x=sweep["threshold"],
                y=sweep[col],
                mode="lines",
                name=col.capitalize(),
                line={"color": color, "width": 2},
            )
        )
    fig.add_vline(
        x=chosen,
        line_dash="dash",
        line_color=PALETTE["neutral"],
        annotation_text=f"trained {chosen:.2f}",
        annotation_position="top",
    )
    if current is not None and abs(current - chosen) > 1e-9:
        fig.add_vline(
            x=current,
            line_dash="dot",
            line_color=PALETTE["red"],
            annotation_text=f"current {current:.2f}",
            annotation_position="bottom",
        )
    fig = _base_layout(fig, "Precision / recall / F1 vs threshold (validation set)")
    fig.update_xaxes(title="Decision threshold", range=[0, 1])
    fig.update_yaxes(range=[0, 1.02])
    return fig


def importance_figure(importance: pd.DataFrame, top_n: int = 15):
    import plotly.graph_objects as go

    s = importance.iloc[:, 0].sort_values(ascending=True).tail(top_n)
    fig = go.Figure(
        go.Bar(
            x=s.values,
            y=s.index,
            orientation="h",
            marker_color=PALETTE["aqua"],
            hovertemplate="%{y}<br>mean |SHAP| = %{x:.3f}<extra></extra>",
        )
    )
    fig = _base_layout(
        fig, "Global feature importance (mean |SHAP|, validation sample)", height=30 * top_n + 80
    )
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title="mean |SHAP value|")
    return fig


def metric_bars_figure(table: pd.DataFrame):
    import plotly.graph_objects as go

    metrics = ["precision", "recall", "f1", "pr_auc", "roc_auc"]
    fig = go.Figure()
    for model in table.index:
        fig.add_trace(
            go.Bar(
                name=model,
                x=metrics,
                y=[table.loc[model, m] for m in metrics],
                marker_color=MODEL_COLORS.get(model, PALETTE["neutral"]),
                hovertemplate="%{x}: %{y:.4f}<extra>" + model + "</extra>",
            )
        )
    fig = _base_layout(
        fig,
        "Model comparison on the untouched test set (each at its own validation-tuned threshold)",
    )
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, hovermode="closest")
    fig.update_yaxes(range=[0, 1.02])
    return fig


def probability_hist_figure(scored: pd.DataFrame, threshold: float):
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Histogram(
            x=scored["fraud_probability"],
            nbinsx=50,
            marker_color=PALETTE["blue"],
            hovertemplate="p=%{x}<br>rows=%{y}<extra></extra>",
        )
    )
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color=PALETTE["red"],
        annotation_text=f"threshold {threshold:.2f}",
    )
    fig = _base_layout(fig, "Distribution of fraud probabilities in this batch", height=300)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title="Fraud probability", range=[0, 1])
    fig.update_yaxes(title="Transactions", type="log")
    return fig
