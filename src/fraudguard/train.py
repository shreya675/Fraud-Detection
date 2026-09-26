"""Train, compare, and select fraud-detection models.

    python -m fraudguard.train

Steps
  1. Load the time-aware splits (``python -m fraudguard.data`` must have run).
  2. Fit the preprocessing pipeline on TRAIN only.
  3. Train Logistic Regression, Random Forest, XGBoost with imbalance handling.
  4. Compare on VALIDATION (PR-AUC is the primary metric).
  5. Choose the decision threshold for the best model on VALIDATION only.
  6. Evaluate once on the untouched TEST set and write reports.
  7. Save preprocessor, model, threshold, feature names and metadata.

Every number written to ``reports/`` and ``models/`` comes out of this run.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraudguard import __version__
from fraudguard.config import (
    DEFAULT_THRESHOLD,
    FEATURE_NAMES_FILE,
    METADATA_FILE,
    MODEL_FILE,
    MODELS_DIR,
    PREPROCESSOR_FILE,
    RANDOM_STATE,
    REPORTS_DIR,
    SAMPLE_DATA_DIR,
    TARGET,
    THRESHOLD_FILE,
)
from fraudguard.data import SUMMARY_FILE, load_splits
from fraudguard.evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_pr_curves,
    plot_roc_curves,
    plot_threshold_sweep,
    select_threshold,
    threshold_sweep,
)
from fraudguard.explain import make_explainer, mean_abs_shap
from fraudguard.features import build_preprocessor, get_feature_names

logger = logging.getLogger(__name__)

MODEL_COMPARISON_FILE = REPORTS_DIR / "model_comparison.json"
MODEL_COMPARISON_CSV = REPORTS_DIR / "model_comparison.csv"
THRESHOLD_ANALYSIS_CSV = REPORTS_DIR / "threshold_analysis.csv"
TEST_METRICS_FILE = REPORTS_DIR / "test_metrics.json"
SAMPLE_FILE = SAMPLE_DATA_DIR / "sample_transactions.csv"


# --------------------------------------------------------------------------- #
# Model factories + hyper-parameter grids
# --------------------------------------------------------------------------- #
# Every model family gets the same treatment: a small grid is scored on the
# validation split (PR-AUC, ties -> widest F1 plateau), and the best config is
# refit on the full training split.  The search runs on a random subsample of
# train (SEARCH_ROWS) to keep it tractable; the refit uses all rows.
SEARCH_ROWS = 1_000_000

LR_GRID: list[dict] = [{"C": c} for c in (0.1, 1.0, 10.0)]

RF_GRID: list[dict] = [
    {"max_depth": d, "min_samples_leaf": leaf} for d in (8, 12, 16) for leaf in (5, 20)
]

# scale_pos_weight labels are resolved against the class ratio at run time
XGB_GRID: list[dict] = [
    {"max_depth": d, "learning_rate": lr, "scale_pos_weight": spw}
    for d in (4, 6, 8)
    for lr in (0.05, 0.1)
    for spw in ("1", "sqrt")
]

QUICK_GRIDS = {  # used by tests / CI
    "logistic_regression": [{"C": 1.0}],
    "random_forest": [{"max_depth": 8, "min_samples_leaf": 5}],
    "xgboost": [{"max_depth": 4, "learning_rate": 0.1, "scale_pos_weight": "1"}],
}


def make_logistic_regression(C: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    C=C,
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_random_forest(
    max_depth: int = 12, min_samples_leaf: int = 20, n_estimators: int = 100
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_samples=0.3,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def make_xgboost(
    scale_pos_weight: float = 1.0,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    n_estimators: int = 500,
) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        objective="binary:logistic",
        # Early stopping on validation log-loss keeps training until the
        # probabilities are well separated (early-stopping on aucpr halts as
        # soon as the *ranking* is perfect, leaving compressed, knife-edge
        # scores). PR-AUC is still tracked and used for model selection.
        eval_metric=["aucpr", "logloss"],
        early_stopping_rounds=50,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


FACTORIES = {
    "logistic_regression": make_logistic_regression,
    "random_forest": make_random_forest,
    "xgboost": make_xgboost,
}


def _params_of(model) -> dict:
    """JSON-serialisable hyper-parameters."""
    est = model.named_steps["clf"] if isinstance(model, Pipeline) else model
    params = est.get_params()
    return {
        k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
        for k, v in params.items()
    }


def _resolve_spw(params: dict, neg: int, pos: int) -> dict:
    """Turn the symbolic scale_pos_weight labels into numbers."""
    out = dict(params)
    spw = out.get("scale_pos_weight")
    if spw == "1":
        out["scale_pos_weight"] = 1.0
    elif spw == "sqrt":
        out["scale_pos_weight"] = float(np.sqrt(neg / pos))
    elif spw == "full":
        out["scale_pos_weight"] = float(neg / pos)
    return out


def _fit(name: str, params: dict, X, y, X_val, y_val, n_estimators: int | None = None):
    kwargs = dict(params)
    if n_estimators is not None:
        kwargs["n_estimators"] = n_estimators
    model = FACTORIES[name](**kwargs)
    if name == "xgboost":
        model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X, y)
    return model


def _score(model, X_val, y_val) -> dict:
    p = model.predict_proba(X_val)[:, 1]
    ap = compute_metrics(y_val, p, DEFAULT_THRESHOLD)["pr_auc"]
    return {
        "validation_pr_auc": round(float(ap), 6),
        "validation_logloss": round(float(log_loss(y_val, p)), 6),
        "validation_f1_plateau_width": f1_plateau_width(y_val, p),
    }


def _config_key(score: dict) -> tuple:
    # PR-AUC first (tie tolerance), then the widest F1 plateau (most robust
    # operating point). Log-loss is recorded but not used: it is not
    # comparable across different class weights.
    return (
        round(score["validation_pr_auc"] / PR_AUC_TIE_TOLERANCE),
        score["validation_f1_plateau_width"],
    )


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_all(
    X_train,
    y_train,
    X_val,
    y_val,
    grids: dict[str, list[dict]] | None = None,
    search_rows: int = SEARCH_ROWS,
    quick: bool = False,
) -> tuple[dict, dict]:
    """Hyper-parameter search + refit for every model family.

    Returns ({name: fitted model}, {name: training info incl. all trials}).
    """
    grids = grids or {"logistic_regression": LR_GRID, "random_forest": RF_GRID, "xgboost": XGB_GRID}
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    n_estimators = {"random_forest": 20, "xgboost": 60} if quick else {}

    # Subsample TRAIN only for the search (validation stays whole)
    if len(X_train) > search_rows:
        idx = np.random.default_rng(RANDOM_STATE).choice(
            len(X_train), size=search_rows, replace=False
        )
        X_search, y_search = X_train.iloc[idx], y_train[idx]
    else:
        X_search, y_search = X_train, y_train
    logger.info(
        "Hyper-parameter search on %s train rows, scored on %s validation rows",
        f"{len(X_search):,}",
        f"{len(X_val):,}",
    )

    models: dict = {}
    info: dict = {}
    for name, grid in grids.items():
        trials, best_key, best_params = [], None, None
        for params in grid:
            resolved = _resolve_spw(params, neg, pos)
            t0 = time.time()
            m = _fit(name, resolved, X_search, y_search, X_val, y_val, n_estimators.get(name))
            score = _score(m, X_val, y_val)
            trial = {"params": params, **score, "search_seconds": round(time.time() - t0, 1)}
            if name == "xgboost":
                trial["best_iteration"] = int(m.best_iteration)
            trials.append(trial)
            logger.info(
                "  %s %s -> val PR-AUC=%.6f plateau=%.2f (%.0fs)",
                name,
                params,
                score["validation_pr_auc"],
                score["validation_f1_plateau_width"],
                time.time() - t0,
            )
            key = _config_key(score)
            if best_key is None or key > best_key:
                best_key, best_params = key, params

        # Refit the winning configuration on the full training split
        t0 = time.time()
        resolved = _resolve_spw(best_params, neg, pos)
        model = _fit(name, resolved, X_train, y_train, X_val, y_val, n_estimators.get(name))
        refit_seconds = round(time.time() - t0, 1)
        models[name] = model
        info[name] = {
            "best_params": best_params,
            "resolved_params": {
                k: (round(v, 3) if isinstance(v, float) else v) for k, v in resolved.items()
            },
            "search_rows": int(len(X_search)),
            "search_trials": trials,
            "search_seconds": round(sum(t["search_seconds"] for t in trials), 1),
            "refit_seconds": refit_seconds,
            "train_seconds": round(sum(t["search_seconds"] for t in trials) + refit_seconds, 1),
            "imbalance": {
                "logistic_regression": "class_weight=balanced",
                "random_forest": "class_weight=balanced_subsample",
                "xgboost": f"scale_pos_weight={best_params.get('scale_pos_weight')} (grid-searched)",
            }[name],
            "params": _params_of(model),
        }
        if name == "xgboost":
            info[name]["best_iteration"] = int(model.best_iteration)
        logger.info(
            "%s: best %s refit on %s rows in %.0fs",
            name,
            best_params,
            f"{len(X_train):,}",
            refit_seconds,
        )
    return models, info


PR_AUC_TIE_TOLERANCE = 5e-4
F1_TIE_TOLERANCE = 5e-4
# Models with a built-in, dependency-free SHAP implementation (requirement: per-transaction explanations)
NATIVE_SHAP_MODELS = {"xgboost"}


def f1_plateau_width(y_true, y_prob) -> float:
    """Width of the threshold range that attains the maximum validation F1.

    A wide plateau means the operating point is robust to small score drift;
    a knife-edge plateau means a tiny shift would change precision/recall.
    """
    sweep = threshold_sweep(y_true, y_prob)
    best = sweep["f1"].max()
    plateau = sweep[sweep["f1"] >= best - 1e-9]["threshold"]
    return round(float(plateau.max() - plateau.min()), 2)


def measure_latency(model, X: pd.DataFrame, n_rows: int = 100_000, repeats: int = 3) -> float:
    """Median wall-clock milliseconds to score ``n_rows`` rows (predict_proba)."""
    sample = X.iloc[: min(n_rows, len(X))]
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict_proba(sample)
        times.append((time.perf_counter() - t0) * 1000)
    return round(float(np.median(times)) * (n_rows / len(sample)), 1)


def evaluate_models(models: dict, X, y, threshold_by_model: dict | None = None) -> dict:
    out = {}
    for name, m in models.items():
        prob = m.predict_proba(X)[:, 1]
        t = (threshold_by_model or {}).get(name, DEFAULT_THRESHOLD)
        out[name] = compute_metrics(y, prob, t)
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(
    splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None = None,
    models_dir: Path = MODELS_DIR,
    reports_dir: Path = REPORTS_DIR,
    sample_dir: Path = SAMPLE_DATA_DIR,
    quick: bool = False,
    deploy: str = "auto",
    search_rows: int = SEARCH_ROWS,
) -> dict:
    """Train/compare/select/evaluate and write all artifacts.

    ``splits`` defaults to the processed PaySim splits on disk; tests pass
    their own small frames.  ``quick=True`` uses one-config grids and tiny
    ensembles so the whole run takes seconds (tests / CI).  ``deploy`` is
    "auto" (rule-based selection) or a model name to deploy by explicit,
    documented decision; the automatic choice is still recorded.
    """
    if deploy not in ("auto", *FACTORIES):
        raise ValueError(f"deploy must be 'auto' or one of {sorted(FACTORIES)}")
    models_dir, reports_dir, sample_dir = Path(models_dir), Path(reports_dir), Path(sample_dir)
    figures_dir = reports_dir / "figures"
    for d in (models_dir, reports_dir, figures_dir, sample_dir):
        d.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    return _run(
        splits,
        models_dir,
        reports_dir,
        figures_dir,
        sample_dir,
        started,
        quick,
        deploy,
        search_rows,
    )


def _run(
    splits,
    models_dir: Path,
    reports_dir: Path,
    figures_dir: Path,
    sample_dir: Path,
    started: datetime,
    quick: bool = False,
    deploy: str = "auto",
    search_rows: int = SEARCH_ROWS,
) -> dict:
    preprocessor_file = models_dir / PREPROCESSOR_FILE.name
    model_file = models_dir / MODEL_FILE.name
    feature_names_file = models_dir / FEATURE_NAMES_FILE.name
    threshold_file = models_dir / THRESHOLD_FILE.name
    metadata_file = models_dir / METADATA_FILE.name
    comparison_file = reports_dir / MODEL_COMPARISON_FILE.name
    comparison_csv = reports_dir / MODEL_COMPARISON_CSV.name
    threshold_csv = reports_dir / THRESHOLD_ANALYSIS_CSV.name
    test_metrics_file = reports_dir / TEST_METRICS_FILE.name
    sample_file = sample_dir / SAMPLE_FILE.name

    train, val, test = splits if splits is not None else load_splits()
    y_train, y_val, y_test = (
        train[TARGET].to_numpy(),
        val[TARGET].to_numpy(),
        test[TARGET].to_numpy(),
    )

    # Fit the preprocessor on TRAIN only
    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(train.drop(columns=[TARGET]))
    X_val = preprocessor.transform(val.drop(columns=[TARGET]))
    X_test = preprocessor.transform(test.drop(columns=[TARGET]))
    feature_names = get_feature_names(preprocessor)
    logger.info(
        "Features: %d | train %s val %s test %s",
        len(feature_names),
        X_train.shape,
        X_val.shape,
        X_test.shape,
    )

    # Train + compare on validation
    models, info = train_all(
        X_train,
        y_train,
        X_val,
        y_val,
        grids=QUICK_GRIDS if quick else None,
        search_rows=min(search_rows, 5_000) if quick else search_rows,
        quick=quick,
    )
    val_prob = {n: m.predict_proba(X_val)[:, 1] for n, m in models.items()}

    val_at_default = {n: compute_metrics(y_val, p, DEFAULT_THRESHOLD) for n, p in val_prob.items()}
    per_model_threshold = {
        n: select_threshold(y_val, p, objective="f1") for n, p in val_prob.items()
    }
    val_at_best = {
        n: compute_metrics(y_val, p, per_model_threshold[n]["threshold"])
        for n, p in val_prob.items()
    }

    # Model selection: highest validation PR-AUC (threshold independent).
    # Models within PR_AUC_TIE_TOLERANCE of the best are considered tied; ties
    # are broken by F1 at the model's own tuned threshold, then by measured
    # inference latency (this is a real-time scoring service).
    latency = {n: measure_latency(m, X_val) for n, m in models.items()}
    for n in models:
        info[n]["inference_ms_per_100k_rows"] = latency[n]
    best_pr_auc = max(val_at_default[n]["pr_auc"] for n in models)
    tied = [n for n in models if val_at_default[n]["pr_auc"] >= best_pr_auc - PR_AUC_TIE_TOLERANCE]
    best_f1 = max(val_at_best[n]["f1"] for n in tied)
    tied = [n for n in tied if val_at_best[n]["f1"] >= best_f1 - F1_TIE_TOLERANCE]
    # Per-transaction SHAP explanations are a system requirement; prefer a
    # model that provides them natively, then the lowest latency.
    best_model_name = min(tied, key=lambda n: (n not in NATIVE_SHAP_MODELS, latency[n]))
    selection_rule = (
        f"highest validation PR-AUC (tie tolerance {PR_AUC_TIE_TOLERANCE}); ties -> highest F1 at tuned threshold "
        f"(tolerance {F1_TIE_TOLERANCE}); ties -> native SHAP support ({sorted(NATIVE_SHAP_MODELS)}); "
        f"ties -> lowest inference latency. Tied set after metric filters: {tied}"
    )
    logger.info("Selection: %s", selection_rule)
    auto_selected = best_model_name
    if deploy != "auto" and deploy != auto_selected:
        selection_rule += (
            f" | MANUAL OVERRIDE: '{deploy}' deployed by explicit decision "
            f"(automatic rule selected '{auto_selected}')"
        )
        best_model_name = deploy
        logger.warning(
            "Deploying %s by manual override (auto rule chose %s)", deploy, auto_selected
        )
    best_model = models[best_model_name]
    chosen = per_model_threshold[best_model_name]
    threshold = chosen["threshold"]
    logger.info(
        "Selected %s (val PR-AUC %.4f); threshold %.2f by %s",
        best_model_name,
        val_at_default[best_model_name]["pr_auc"],
        threshold,
        chosen["rule"],
    )

    # Threshold analysis (validation only)
    sweep = threshold_sweep(y_val, val_prob[best_model_name])
    sweep.to_csv(threshold_csv, index=False)
    plot_threshold_sweep(sweep, threshold, best_model_name, figures_dir / "threshold_analysis.png")

    # ---- Final evaluation on the untouched TEST set -------------------- #
    test_prob = {n: m.predict_proba(X_test)[:, 1] for n, m in models.items()}
    test_metrics_all = {
        n: compute_metrics(y_test, p, per_model_threshold[n]["threshold"])
        for n, p in test_prob.items()
    }
    test_metrics = test_metrics_all[best_model_name]
    test_metrics_default = compute_metrics(y_test, test_prob[best_model_name], DEFAULT_THRESHOLD)

    # Figures
    plot_pr_curves(
        {n: (y_val, p) for n, p in val_prob.items()},
        "Precision-Recall curves (validation)",
        figures_dir / "pr_curve_validation.png",
    )
    plot_pr_curves(
        {n: (y_test, p) for n, p in test_prob.items()},
        "Precision-Recall curves (test)",
        figures_dir / "pr_curve_test.png",
    )
    plot_roc_curves(
        {n: (y_test, p) for n, p in test_prob.items()},
        "ROC curves (test)",
        figures_dir / "roc_curve_test.png",
    )
    plot_confusion_matrix(
        test_metrics,
        f"Confusion matrix (test) - {best_model_name} @ {threshold:.2f}",
        figures_dir / "confusion_matrix_test.png",
    )

    # Global SHAP importance (computed on a validation sample - never on test)
    importance = None
    explainer = make_explainer(
        best_model_name,
        best_model,
        best_model.get_booster() if best_model_name == "xgboost" else None,
    )
    if explainer is not None:
        n_sample = (
            50_000 if best_model_name == "xgboost" else 5_000
        )  # shap.TreeExplainer on a forest is slower
        sample = X_val.sample(n=min(n_sample, len(X_val)), random_state=RANDOM_STATE)
        importance = mean_abs_shap(explainer, sample)
        plot_feature_importance(
            importance,
            "Global feature importance (mean |SHAP|, validation sample)",
            figures_dir / "shap_feature_importance.png",
        )
        importance.to_csv(reports_dir / "shap_feature_importance.csv", header=["mean_abs_shap"])

    # ---- Persist artifacts --------------------------------------------- #
    joblib.dump(preprocessor, preprocessor_file)
    if best_model_name == "xgboost":
        # Persist the raw Booster sliced to the early-stopping best iteration.
        # (Avoids the sklearn-wrapper save path, which depends on the exact
        # xgboost/scikit-learn version pairing.)
        booster = best_model.get_booster()[: int(best_model.best_iteration) + 1]
        booster.save_model(str(model_file))
        model_path = model_file
    else:
        model_path = models_dir / f"{best_model_name}.joblib"
        joblib.dump(best_model, model_path)
    feature_names_file.write_text(json.dumps(feature_names, indent=2))
    threshold_file.write_text(
        json.dumps({**chosen, "default_threshold": DEFAULT_THRESHOLD}, indent=2)
    )

    comparison = {
        "primary_metric": "validation pr_auc",
        "selection_rule": selection_rule,
        "auto_selected_model": auto_selected,
        "selected_model": best_model_name,
        "models": {
            n: {
                "training": info[n],
                "validation_at_0.5": val_at_default[n],
                "validation_threshold": per_model_threshold[n],
                "validation_at_own_threshold": val_at_best[n],
                "test_at_own_threshold": test_metrics_all[n],
            }
            for n in models
        },
    }
    comparison_file.write_text(json.dumps(comparison, indent=2))
    rows = []
    for n in models:
        for split, m in [("validation", val_at_best[n]), ("test", test_metrics_all[n])]:
            rows.append(
                {
                    "model": n,
                    "split": split,
                    **{
                        k: m[k]
                        for k in (
                            "threshold",
                            "precision",
                            "recall",
                            "f1",
                            "pr_auc",
                            "roc_auc",
                            "tp",
                            "fp",
                            "fn",
                            "tn",
                        )
                    },
                }
            )
    pd.DataFrame(rows).to_csv(comparison_csv, index=False)

    test_metrics_file.write_text(
        json.dumps(
            {
                "model": best_model_name,
                "at_selected_threshold": test_metrics,
                "at_default_threshold_0.5": test_metrics_default,
            },
            indent=2,
        )
    )

    data_summary = (
        json.loads(SUMMARY_FILE.read_text()) if (splits is None and SUMMARY_FILE.exists()) else {}
    )
    metadata = {
        "project": "FraudGuard",
        "version": __version__,
        "model_type": best_model_name,
        "model_file": model_path.name,
        "trained_at_utc": started.isoformat(),
        "training_seconds_total": round((datetime.now(timezone.utc) - started).total_seconds(), 1),
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "xgboost": xgb.__version__,
            "pandas": pd.__version__,
        },
        "dataset": data_summary.get("dataset"),
        "splits": data_summary.get("splits"),
        "split_strategy": data_summary.get("split_strategy"),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "hyperparameters": info[best_model_name],
        "selection_rule": selection_rule,
        "auto_selected_model": auto_selected,
        "threshold": chosen,
        "validation_metrics": val_at_best[best_model_name],
        "test_metrics": test_metrics,
        "top_features_mean_abs_shap": importance.head(10).round(4).to_dict()
        if importance is not None
        else None,
    }
    metadata_file.write_text(json.dumps(metadata, indent=2))

    # A small demo file for batch scoring, drawn from TEST (never seen in training)
    demo = test.sample(n=min(500, len(test)), random_state=RANDOM_STATE)
    demo = pd.concat(
        [
            demo,
            test[test[TARGET] == 1].sample(
                n=min(25, int((test[TARGET] == 1).sum())), random_state=RANDOM_STATE
            ),
        ]
    ).drop_duplicates()
    demo.sample(frac=1, random_state=RANDOM_STATE).to_csv(sample_file, index=False)

    logger.info("Artifacts written to %s and %s", models_dir, reports_dir)
    return metadata


def _cli() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Train, compare and select FraudGuard models")
    ap.add_argument(
        "--deploy",
        default="auto",
        choices=["auto", *sorted(FACTORIES)],
        help="model to deploy (default: rule-based selection)",
    )
    ap.add_argument(
        "--search-rows",
        type=int,
        default=SEARCH_ROWS,
        help="train rows used for the hyper-parameter search",
    )
    ap.add_argument("--quick", action="store_true", help="tiny grids/ensembles (smoke test)")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout
    )
    meta = run(quick=args.quick, deploy=args.deploy, search_rows=args.search_rows)
    print(
        json.dumps(
            {
                k: meta[k]
                for k in (
                    "model_type",
                    "auto_selected_model",
                    "threshold",
                    "validation_metrics",
                    "test_metrics",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    _cli()
