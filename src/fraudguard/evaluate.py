"""Metrics, threshold analysis, and report figures.

All functions are pure (arrays in, numbers/frames out) so they are trivially
testable and reusable by the dashboard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# Chart palette
COLORS = {
    "logistic_regression": "#2a78d6",  # blue
    "random_forest": "#eb6834",  # orange
    "xgboost": "#1baf7a",  # aqua
    "fraud": "#e34948",  # status red
    "legit": "#2a78d6",
    "neutral": "#52514e",
    "grid": "#e6e5e1",
}


# --------------------------------------------------------------------------- #
# Core metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_prob, threshold: float) -> dict:
    """Classification metrics at a given probability threshold.

    Returns precision, recall, f1, PR-AUC (average precision), ROC-AUC and the
    raw confusion-matrix counts (tp, fp, fn, tn).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same shape")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be within [0, 1]")

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    has_both_classes = 0 < y_true.sum() < len(y_true)
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(average_precision_score(y_true, y_prob))
        if has_both_classes
        else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if has_both_classes else float("nan"),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
    }


def threshold_sweep(y_true, y_prob, thresholds=None) -> pd.DataFrame:
    """Precision / recall / F1 / FP / FN for every candidate threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)

    positives = y_true.sum()
    order = np.argsort(-y_prob)
    sorted_true = y_true[order]
    sorted_prob = y_prob[order]
    cum_tp = np.cumsum(sorted_true)
    cum_fp = np.cumsum(1 - sorted_true)

    rows = []
    for t in thresholds:
        k = int(np.searchsorted(-sorted_prob, -t, side="right"))  # count of prob >= t
        tp = int(cum_tp[k - 1]) if k else 0
        fp = int(cum_fp[k - 1]) if k else 0
        fn = int(positives - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / positives if positives else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "threshold": float(t),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "flagged": tp + fp,
            }
        )
    return pd.DataFrame(rows)


def select_threshold(
    y_true, y_prob, objective: str = "f1", min_precision: float | None = None
) -> dict:
    """Pick the decision threshold on *validation* data.

    objective = "f1"        -> maximise F1
    min_precision = 0.9     -> maximise recall subject to precision >= 0.9
    """
    sweep = threshold_sweep(y_true, y_prob)
    if min_precision is not None:
        candidates = sweep[sweep["precision"] >= min_precision]
        if candidates.empty:
            raise ValueError(f"No threshold reaches precision >= {min_precision}")
        best = candidates.sort_values(["recall", "f1"], ascending=False).iloc[0]
        rule = f"max recall with precision >= {min_precision}"
    elif objective == "f1":
        # All thresholds within a hair of the best F1 form a plateau; take its
        # midpoint so the operating point is robust to small score drift.
        best_f1 = sweep["f1"].max()
        plateau = sweep[sweep["f1"] >= best_f1 - 1e-9]
        mid = float(plateau["threshold"].iloc[len(plateau) // 2])
        best = sweep.loc[sweep["threshold"] == mid].iloc[0]
        rule = (
            f"max F1 on validation (plateau {plateau['threshold'].min():.2f}-{plateau['threshold'].max():.2f}, midpoint chosen)"
            if len(plateau) > 1
            else "max F1 on validation"
        )
    else:
        raise ValueError(f"Unknown objective: {objective}")
    return {
        "threshold": float(best["threshold"]),
        "rule": rule,
        "validation_precision": float(best["precision"]),
        "validation_recall": float(best["recall"]),
        "validation_f1": float(best["f1"]),
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.grid(True, color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=COLORS["neutral"])


def plot_pr_curves(curves: dict[str, tuple], title: str, path: Path) -> None:
    """curves = {model_name: (y_true, y_prob)}"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5), dpi=130)
    for name, (y_true, y_prob) in curves.items():
        p, r, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(
            r,
            p,
            lw=2,
            color=COLORS.get(name, COLORS["neutral"]),
            label=f"{name} (PR-AUC = {ap:.3f})",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title, loc="left", fontsize=12, color=COLORS["neutral"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_roc_curves(curves: dict[str, tuple], title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5), dpi=130)
    for name, (y_true, y_prob) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(
            fpr,
            tpr,
            lw=2,
            color=COLORS.get(name, COLORS["neutral"]),
            label=f"{name} (ROC-AUC = {auc:.4f})",
        )
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=COLORS["grid"])
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, loc="left", fontsize=12, color=COLORS["neutral"])
    ax.legend(frameon=False, loc="lower right")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_threshold_sweep(sweep: pd.DataFrame, chosen: float, model_name: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5), dpi=130)
    ax.plot(
        sweep["threshold"],
        sweep["precision"],
        lw=2,
        color=COLORS["logistic_regression"],
        label="Precision",
    )
    ax.plot(
        sweep["threshold"], sweep["recall"], lw=2, color=COLORS["random_forest"], label="Recall"
    )
    ax.plot(sweep["threshold"], sweep["f1"], lw=2, color=COLORS["xgboost"], label="F1")
    ax.axvline(chosen, color=COLORS["neutral"], ls="--", lw=1)
    ax.annotate(
        f"chosen = {chosen:.2f}",
        xy=(chosen, 0.05),
        xytext=(5, 0),
        textcoords="offset points",
        color=COLORS["neutral"],
        fontsize=9,
    )
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title(
        f"Threshold analysis on validation - {model_name}",
        loc="left",
        fontsize=12,
        color=COLORS["neutral"],
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion_matrix(metrics: dict, title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4.2), dpi=130)
    ax.imshow(np.log1p(cm), cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(
            j,
            i,
            f"{v:,}",
            ha="center",
            va="center",
            fontsize=12,
            color="#0b0b0b" if np.log1p(v) < np.log1p(cm).max() * 0.6 else "white",
        )
    ax.set_xticks([0, 1], ["Predicted legit", "Predicted fraud"])
    ax.set_yticks([0, 1], ["Actual legit", "Actual fraud"])
    ax.set_title(title, loc="left", fontsize=11, color=COLORS["neutral"])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_feature_importance(importance: pd.Series, title: str, path: Path, top_n: int = 15) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top = importance.sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=130)
    ax.barh(top.index, top.values, color=COLORS["xgboost"], height=0.6)
    ax.set_title(title, loc="left", fontsize=12, color=COLORS["neutral"])
    ax.set_xlabel("Mean |SHAP value| (impact on model output)")
    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
